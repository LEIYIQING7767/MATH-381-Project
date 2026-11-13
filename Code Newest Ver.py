#!/usr/bin/env python3
"""Build course times and student schedules for the MathILy-Er camp.

Run these commands from the folder with this script and the survey.

    python -m pip install ortools==9.15.6755
    python mathily_er_scheduler_beginner.py survey.txt --out results

The survey has one column per course. The first three rows give the
instructor, class name, and topic. Every other row gives a student's
ratings. S means strongly preferred, L means liked, blank means
acceptable, and X means the student cannot take the course.

Open START_HERE.txt for a summary. The two CSV files show the schedule
by student and by course. Choose a new output folder for each run.

If you are reading the code for the first time, start at main() at the
bottom. It follows the same order as a run.
"""

import argparse
import csv
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache
from itertools import combinations
from pathlib import Path
from time import perf_counter

try:
    from ortools.sat.python import cp_model
except ImportError:
    raise SystemExit("Install OR-Tools with: python -m pip install ortools==9.15.6755")


# 1. Settings staff can change.

# Each number says how many courses meet at the same time in that slot.
SLOT_SIZES = (4, 4, 4, 3, 3)
MAX_SAME_TOPIC = 3  # A student can take at most this many courses in one known topic.
MAX_CLASS_SIZE = None  # With None, a class can have up to the total number of students.

# Use the exact course names from the survey. Remove # to use an example line.
TOPIC_OVERRIDES = {
    # "Exact course title": "Algebra",
}
STUDENT_COURSE_RULES = [
    # ("Student 2", "Algebraists Anonymous", True),  # This student must take it.
    # ("Student 2", "Complaints in Noir Films", False),  # This student cannot take it.
]
COURSE_SLOT_RULES = {
    # "Algebraists Anonymous": 2,  # This course must meet in slot 2.
}


# 2. Read the survey and turn the manual rules into course and student numbers.

# These small classes keep related facts together, like the fields on a course card.
@dataclass(frozen=True)
class Course:
    name: str
    instructor: str
    topic: str


@dataclass(frozen=True)
class Student:
    name: str
    gender: str
    ratings: tuple[str, ...]  # Kept in the same order as the courses in the survey.


@dataclass
class Locks:
    # These are the course times and student choices that staff have fixed.
    course_slots: dict[int, int]
    student_courses: dict[tuple[int, int], bool]


def read_survey(path):
    """Read and check the course columns and student rows."""
    separator = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open(encoding="utf-8-sig", newline="") as file:
        # Skip entirely blank rows. Keep blank cells inside the other rows.
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(file, delimiter=separator)
            if any(cell.strip() for cell in row)
        ]

    headers = [row[0].casefold() for row in rows[:3]]
    if len(rows) < 4 or headers != ["instructor", "class name", "topic"]:
        raise ValueError("Expected Instructor, Class name, and Topic in the first three rows.")

    # The first cell is the row label. An export can have extra blank columns at the end.
    names = rows[1][1:]
    while names and not names[-1]:
        names.pop()
    if len(names) != 18 or not all(names) or len(set(names)) != 18:
        raise ValueError("Expected 18 different course names with no gaps.")

    def course_cells(row):
        if any(row[19:]):
            raise ValueError(f"Unexpected values past the last course in row {row[0]!r}.")
        cells = row[1:19]
        # Some exports drop empty cells at the end. Put them back in their columns.
        return cells + [""] * (18 - len(cells))

    instructors = course_cells(rows[0])
    if not all(instructors):
        raise ValueError("Every course needs an instructor.")
    unknown_topics = TOPIC_OVERRIDES.keys() - set(names)
    if unknown_topics:
        raise ValueError(f"Unknown course in TOPIC_OVERRIDES: {', '.join(sorted(unknown_topics))}.")

    courses = []
    for name, instructor, topic in zip(names, instructors, course_cells(rows[2])):
        confirmed_topic = TOPIC_OVERRIDES.get(name, topic).strip()
        courses.append(Course(name, instructor, confirmed_topic))

    students = []
    names_seen = set()
    for row in rows[3:]:
        # Split "Student 2 (M)" into the name and the label in parentheses.
        match = re.fullmatch(r"(.*?)\s*\(([^()]*)\)", row[0])
        name = match[1].strip() if match else row[0]
        gender = match[2].strip().upper() if match else ""
        if gender == "W":
            gender = "F"
        if not name or name in names_seen:
            raise ValueError(f"Missing or repeated student name in row {row[0]!r}.")
        names_seen.add(name)
        # Some copied survey letters carry an invisible display mark. Remove it.
        ratings = tuple(re.sub(r"[\ufe0e\ufe0f]", "", cell).upper() for cell in course_cells(row))
        for course, rating in zip(courses, ratings):
            if rating not in ("S", "L", "", "X"):
                raise ValueError(f"Unknown rating {rating!r} for {name}, {course.name!r}.")
        students.append(Student(name, gender, ratings))
    return courses, students


def make_locks(courses, students):
    """Use names in the settings and list positions in the model."""
    course_ids = {course.name: number for number, course in enumerate(courses)}
    student_ids = {student.name: number for number, student in enumerate(students)}
    locks = Locks({}, {})
    for name, slot in COURSE_SLOT_RULES.items():
        if name not in course_ids:
            raise ValueError(f"Unknown course in COURSE_SLOT_RULES: {name!r}.")
        if type(slot) is not int or not 1 <= slot <= len(SLOT_SIZES):
            raise ValueError(f"Slot for {name!r} must be an integer from 1 to {len(SLOT_SIZES)}.")
        # Staff call the first slot 1. Python calls the first list position 0.
        locks.course_slots[course_ids[name]] = slot - 1

    for name, course, required in STUDENT_COURSE_RULES:
        if name not in student_ids or course not in course_ids or type(required) is not bool:
            raise ValueError(f"Check the student name, course name, and True/False for {name}, {course!r}.")
        student_id, course_id = student_ids[name], course_ids[course]
        if required and students[student_id].ratings[course_id] == "X":
            raise ValueError(f"{name} rated {course!r} X and cannot be required to take it.")
        pair = (student_id, course_id)
        if pair in locks.student_courses and locks.student_courses[pair] != required:
            raise ValueError(f"Conflicting rules for {name} and {course!r}.")
        locks.student_courses[pair] = required
    return locks


# 3. Tell the solver which choices it can make and which rules it must follow.
#
# We want whole choices. Algebra is in slot 2 or it is not. A student takes
# Algebra or does not. If we set course times first, two popular courses may
# meet together and their students will have to choose between them.
# CP-SAT lets us ask both sets of questions together using 0 and 1.
# It can drop a partial schedule as soon as it breaks a rule, such as
# putting one teacher in two places. That saves us from listing every full schedule.
# Python describes the rules. OR-Tools searches and reports what it can prove.
# https://developers.google.com/optimization/cp/cp_solver

@dataclass
class ScheduleModel:
    model: cp_model.CpModel
    takes_course: list[list[cp_model.IntVar]]
    course_time: list[cp_model.IntVar]
    strong_choices: cp_model.LinearExpr | int
    liked_choices: cp_model.LinearExpr | int
    goals: list[tuple[str, cp_model.LinearExpr | int, bool]]


def build_model(courses, students, slot_sizes, locks, max_class_size):
    model = cp_model.CpModel()
    course_ids = range(len(courses))
    student_ids = range(len(students))
    slots = range(len(slot_sizes))
    instructors, topics = defaultdict(list), defaultdict(list)
    for course_id, course in enumerate(courses):
        instructors[course.instructor].append(course_id)
        if course.topic:
            topics[course.topic.casefold()].append(course_id)

    # course_at[c][t] is 1 if course c meets in slot t.
    course_at = [
        [model.new_bool_var(f"course_{c}_in_slot_{t}") for t in slots]
        for c in course_ids
    ]
    # takes_course[s][c] is 1 if student s takes course c in any slot.
    takes_course = [
        [model.new_bool_var(f"student_{s}_takes_course_{c}") for c in course_ids]
        for s in student_ids
    ]
    course_time = [model.new_int_var(0, len(slots) - 1, f"slot_of_course_{c}") for c in course_ids]

    # We want one time for each course, then the right number of courses per time.
    for c in course_ids:
        model.add_exactly_one(course_at[c])
        # Only the chosen slot has a 1. The sum returns that slot's number.
        model.add(course_time[c] == sum(t * course_at[c][t] for t in slots))
    for t in slots:
        model.add(sum(course_at[c][t] for c in course_ids) == slot_sizes[t])
        # A teacher can be in at most one place at this time.
        for teacher_courses in instructors.values():
            model.add_at_most_one(course_at[c][t] for c in teacher_courses)
    for c, t in locks.course_slots.items():
        model.add(course_at[c][t] == 1)

    # Two slots with the same number of courses and no fixed times can swap.
    # We search one order by putting the smaller first course earlier.
    for size in set(slot_sizes):
        free_slots = [t for t in slots if slot_sizes[t] == size and t not in locks.course_slots.values()]
        first_courses = []
        for t in free_slots:
            first = model.new_int_var(0, len(courses) - 1, f"first_course_in_slot_{t}")
            # Add a large number to courses outside this slot so they cannot be first.
            model.add_min_equality(first, [c + len(courses) * (1 - course_at[c][t]) for c in course_ids])
            first_courses.append(first)
        for earlier, later in zip(first_courses, first_courses[1:]):
            model.add(earlier < later)

    schedule_numbers = []
    for s, student in enumerate(students):
        choices_by_slot = [[] for _ in slots]
        for c, rating in enumerate(student.ratings):
            staff_rule = locks.student_courses.get((s, c))
            if rating == "X" or staff_rule is False:
                model.add(takes_course[s][c] == 0)
                continue
            if staff_rule is True:
                model.add(takes_course[s][c] == 1)

            # We need to link what the student takes to when that course meets.
            # If course_at is 0 in this slot, the student's choice must also be 0.
            times_to_check = slots if c not in locks.course_slots else [locks.course_slots[c]]
            choices_for_course = []
            for t in times_to_check:
                choice = model.new_bool_var(f"student_{s}_takes_course_{c}_in_slot_{t}")
                model.add(choice <= course_at[c][t])
                choices_for_course.append(choice)
                choices_by_slot[t].append(choice)
            model.add(sum(choices_for_course) == takes_course[s][c])

        # Each student must go to one course in every slot.
        for choices in choices_by_slot:
            model.add_exactly_one(choices)
        model.add(sum(takes_course[s]) == len(slots))

        # Give the courses numbers 1, 2, 4, 8, and so on.
        # Each number is bigger than all earlier ones added together.
        # Adding the chosen numbers identifies the student's exact set of courses.
        number = model.new_int_var(0, 2 ** len(courses) - 1, f"schedule_number_{s}")
        model.add(number == sum(2 ** c * takes_course[s][c] for c in course_ids))
        schedule_numbers.append(number)
    model.add_all_different(schedule_numbers)

    strong_choices, liked_choices, goals = make_scores(
        model, courses, students, slot_sizes, takes_course, course_at,
        instructors, topics, max_class_size,
    )
    return ScheduleModel(model, takes_course, course_time, strong_choices, liked_choices, goals)


# 4. Give each legal schedule a set of scores we can compare.

def make_scores(model, courses, students, slot_sizes, takes_course, course_at,
                instructors, topics, max_class_size):
    course_ids = range(len(courses))
    student_ids = range(len(students))
    slots = range(len(slot_sizes))
    total_seats = len(students) * len(slots)

    # An S adds one to the S count only when we assign that student that course.
    strong_choices = sum(takes_course[s][c] for s in student_ids for c in course_ids
                         if students[s].ratings[c] == "S")
    liked_choices = sum(takes_course[s][c] for s in student_ids for c in course_ids
                        if students[s].ratings[c] == "L")
    # With 160 seats, one S is worth 161 points and one L is worth 1.
    # This keeps a single extra S ahead of any possible gain in L.
    preference_score = (total_seats + 1) * strong_choices + liked_choices

    group_f = [s for s in student_ids if students[s].gender == "F"]
    group_m = [s for s in student_ids if students[s].gender == "M"]
    largest_gap = max(len(group_f), len(group_m))
    square_of_gap = [gap ** 2 for gap in range(largest_gap + 1)]
    gender_scores, size_scores = [], []
    for c in course_ids:
        enrollment = sum(takes_course[s][c] for s in student_ids)
        # A course must have someone in it. Staff can also set an upper limit.
        model.add_linear_constraint(enrollment, 1, max_class_size)

        # Count the F and M labels in this course. A gap of 3 costs 9 points.
        # Squaring makes a big gap cost more than several small ones.
        count_f = sum(takes_course[s][c] for s in group_f)
        count_m = sum(takes_course[s][c] for s in group_m)
        gap = model.new_int_var(0, largest_gap, f"gender_gap_{c}")
        model.add_abs_equality(gap, count_m - count_f)
        squared_gap = model.new_int_var(0, largest_gap ** 2, f"gender_points_{c}")
        # Look up the square in the list [0, 1, 4, 9, ...].
        model.add_element(gap, square_of_gap, squared_gap)
        gender_scores.append(squared_gap)

        # For 32 students in a 3-course slot, aim for 10 or 11 students per course.
        # The chosen slot sets the target. Inside the range the penalty is zero.
        low = sum((len(students) // slot_sizes[t]) * course_at[c][t] for t in slots)
        high = sum(math.ceil(len(students) / slot_sizes[t]) * course_at[c][t] for t in slots)
        size_penalty = model.new_int_var(0, len(students), f"size_points_{c}")
        # Of these three numbers, only the one for a shortage or excess can be positive.
        model.add_max_equality(size_penalty, [0, low - enrollment, enrollment - high])
        size_scores.append(size_penalty)

    # Each slot contains every student. That fixes its total M-minus-F difference.
    # Spreading that difference as evenly as possible gives a safe minimum penalty.
    min_gender_score = 0
    for course_count in slot_sizes:
        gap, courses_with_extra = divmod(len(group_m) - len(group_f), course_count)
        min_gender_score += (course_count - courses_with_extra) * gap ** 2
        min_gender_score += courses_with_extra * (gap + 1) ** 2
    model.add(sum(gender_scores) >= min_gender_score)

    def count_repeats(groups, label, cap=None):
        """The first course in a group is free. Each additional one costs a point."""
        penalties = []
        for s in student_ids:
            for group_number, group in enumerate(groups.values()):
                if len(group) <= 1:
                    continue
                courses_taken = sum(takes_course[s][c] for c in group)
                # For topics the limit is a rule even if breaking it would improve preferences.
                if cap is not None and len(group) > cap:
                    model.add(courses_taken <= cap)
                repeats = model.new_int_var(0, len(group) - 1, f"{label}_repeats_{s}_{group_number}")
                model.add_max_equality(repeats, [0, courses_taken - 1])
                penalties.append(repeats)
        return sum(penalties)

    # The last field says whether bigger is better. The list order is the priority.
    goals = [
        ("Student preferences", preference_score, True),
        ("Gender balance", sum(gender_scores), False),
        ("Topic variety", count_repeats(topics, "topic", MAX_SAME_TOPIC), False),
        ("Instructor variety", count_repeats(instructors, "instructor"), False),
        ("Class sizes", sum(size_scores), False),
    ]
    return strong_choices, liked_choices, goals


# 5. Once we know the preference score to keep, skip timetables that cannot reach it.
# This is a speedup. On a first read, skip to section 6 for the main search.

def narrow_timetable_search(built, courses, students, slot_sizes, score_to_keep, limit=200):
    """Use a generous estimate so we only rule out choices that cannot work."""
    course_ids = range(len(courses))
    points = {"S": len(students) * len(slot_sizes) + 1, "L": 1, "": 0, "X": -1}
    ratings = [[points[rating] for rating in student.ratings] for student in students]

    # Imagine every student gets their favorite course in the slot.
    # That can overestimate the score, which is safe for ruling out bad slots.
    groups_by_first_course = [[] for _ in courses]
    for size in sorted(set(slot_sizes)):
        for group in combinations(course_ids, size):
            if len({courses[c].instructor for c in group}) < size:
                continue  # One teacher would have to be in two places.
            best_ratings = [max(row[c] for c in group) for row in ratings]
            if min(best_ratings) >= 0:  # Everyone has at least one choice they did not rate X.
                groups_by_first_course[group[0]].append((frozenset(group), sum(best_ratings)))

    def next_groups(courses_left, threes_left, fours_left):
        # Every complete timetable puts the first remaining course somewhere.
        # Pick its group first so we do not build the same groups in many orders.
        for group, score in groups_by_first_course[min(courses_left)]:
            next_threes = threes_left - (len(group) == 3)
            next_fours = fours_left - (len(group) == 4)
            if next_threes >= 0 and next_fours >= 0 and group <= courses_left:
                yield group, score, courses_left - group, next_threes, next_fours

    @cache
    def best_remaining_score(courses_left, threes_left, fours_left):
        # When two paths leave the same courses and slots, reuse the answer.
        if not courses_left:
            return 0
        best_score = -math.inf
        for _, score, rest, threes, fours in next_groups(courses_left, threes_left, fours_left):
            best_score = max(best_score, score + best_remaining_score(rest, threes, fours))
        return best_score

    timetables = []

    def collect_timetables(courses_left, threes_left, fours_left, points_needed, groups):
        if not courses_left:
            # These groups arrived in order of their first course. Use that same
            # order within each slot size so the timetable matches the model.
            groups_by_size = {size: iter([g for g in groups if len(g) == size])
                              for size in set(slot_sizes)}
            timetable = [0] * len(courses)
            for t, size in enumerate(slot_sizes):
                for c in next(groups_by_size[size]):
                    timetable[c] = t
            timetables.append(tuple(timetable))
            return

        for group, score, rest, threes, fours in next_groups(courses_left, threes_left, fours_left):
            # If even the generous estimate falls short, there is no reason to go on.
            if score + best_remaining_score(rest, threes, fours) >= points_needed:
                collect_timetables(rest, threes, fours, points_needed - score, [*groups, group])
                if len(timetables) > limit:
                    return

    collect_timetables(frozenset(course_ids), slot_sizes.count(3), slot_sizes.count(4), score_to_keep, [])
    if len(timetables) > limit:
        return  # A long list is costly to add. Let the solver use its original rules.
    if not timetables:
        raise RuntimeError("The timetable check lost a schedule that had already passed the rules.")
    built.model.add_allowed_assignments(built.course_time, timetables)

    # A timetable with an estimate of 105 can lose only 5 points and still get 100.
    # A student choice that loses 6 points cannot help in that timetable.
    # Keep a student-course choice if at least one timetable can afford the loss.
    possible = [[False] * len(courses) for _ in students]
    for timetable in timetables:
        courses_by_slot = [[c for c, time in enumerate(timetable) if time == t]
                           for t in range(len(slot_sizes))]
        best = [[max(row[c] for c in group) for group in courses_by_slot] for row in ratings]
        points_we_can_lose = sum(map(sum, best)) - score_to_keep
        for s, row in enumerate(ratings):
            for c, rating in enumerate(row):
                if rating >= 0 and best[s][timetable[c]] - rating <= points_we_can_lose:
                    possible[s][c] = True
    for s, choices in enumerate(possible):
        for c, can_take in enumerate(choices):
            if not can_take:
                built.model.add(built.takes_course[s][c] == 0)
    print(f"  {len(timetables)} course timetables pass the preference score check.", flush=True)


# 6. Improve one goal at a time and keep the score before moving on.
# We want better class sizes only if we can keep the earlier scores,
# including the counts of students' preferred courses.

@dataclass(frozen=True)
class SearchRound:
    name: str
    score: int
    higher_is_better: bool
    best_possible_bound: int | None
    status: str
    proved_through_this_round: bool
    seconds: float


def solve_in_priority_order(built, courses, students, slot_sizes, locks, seconds, workers, seed):
    """Use the best schedule we have after each round as the next starting point."""
    model = built.model
    best = None
    rounds = []
    earlier_rounds_proved = True
    variables = [model.get_int_var_from_proto_index(i) for i in range(len(model.proto.variables))]

    for number, (name, score, higher_is_better) in enumerate(built.goals, start=1):
        print(f"Round {number} of {len(built.goals)} — {name}", flush=True)
        if higher_is_better:
            model.maximize(score)
        else:
            model.minimize(score)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds
        solver.parameters.num_workers = workers
        solver.parameters.random_seed = seed
        status = solver.solve(model)
        if status == cp_model.MODEL_INVALID:
            raise RuntimeError(f"The solver rejected the model. {model.validate() or solver.solution_info()}")
        if status == cp_model.INFEASIBLE:
            if best is None:
                raise ValueError("The solver proved no schedule can satisfy all the current rules.")
            raise RuntimeError("A later round lost a schedule that had already passed the rules.")
        if status == cp_model.UNKNOWN and best is None:
            raise ValueError("No schedule was found in time. Try a larger --seconds-per-stage value.")

        # A later round can run out of time. Keep the previous schedule if needed.
        bound = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            bound = round(solver.best_objective_bound)
            choose_better = max if higher_is_better else min
            found = solver.value(score)
            if best is None or found == choose_better(found, best.value(score)):
                best = solver

        score_to_keep = best.value(score)
        # A proof for this round assumes the earlier scores we kept.
        # We can claim the full sequence through this round only if they were proved too.
        earlier_rounds_proved = earlier_rounds_proved and status == cp_model.OPTIMAL
        rounds.append(SearchRound(
            name, score_to_keep, higher_is_better, bound, solver.status_name(status),
            earlier_rounds_proved, solver.wall_time,
        ))
        if status == cp_model.OPTIMAL:
            message = "Proved best with earlier scores fixed."
        elif status == cp_model.FEASIBLE:
            message = "Found a legal schedule. A better score may be possible."
        else:
            message = "No new schedule found in time. Kept the earlier schedule."
        print(f"  Kept score {score_to_keep}. {message}", flush=True)

        if number == 1:
            # Keep both survey counts, so later rounds cannot trade an S for more Ls.
            model.add(built.strong_choices == best.value(built.strong_choices))
            model.add(built.liked_choices == best.value(built.liked_choices))
            if not locks.course_slots:
                narrow_timetable_search(built, courses, students, slot_sizes, score_to_keep)
        else:
            model.add(score == score_to_keep)

        # Give the solver a schedule it can start from on the next goal.
        model.clear_hints()
        for variable in variables:
            model.add_hint(variable, best.value(variable))
    return best, rounds


# 7. Check the chosen schedule before writing any files.

def check_schedule(courses, students, slot_sizes, locks, max_class_size, course_times, schedules):
    """Check the result directly, using course times and student lists."""
    if Counter(course_times) != Counter(dict(enumerate(slot_sizes))):
        raise RuntimeError("The timetable has the wrong number of courses in a slot.")
    for t in range(len(slot_sizes)):
        teachers = [courses[c].instructor for c, time in enumerate(course_times) if time == t]
        if len(teachers) != len(set(teachers)):
            raise RuntimeError("An instructor has two courses at the same time.")
    if len(schedules) != len(students) or len(set(schedules)) != len(students):
        raise RuntimeError("A student is missing a schedule or two schedules are the same.")

    enrollment = Counter()
    for s, schedule in enumerate(schedules):
        if len(schedule) != len(slot_sizes) or any(course_times[c] != t for t, c in enumerate(schedule)):
            raise RuntimeError("A student does not have one course in every slot.")
        if any(students[s].ratings[c] == "X" for c in schedule):
            raise RuntimeError("A student was assigned a course they rated X.")
        topics = Counter(courses[c].topic.casefold() for c in schedule if courses[c].topic)
        if any(count > MAX_SAME_TOPIC for count in topics.values()):
            raise RuntimeError("A student has too many courses in one topic.")
        enrollment.update(schedule)
    if any(not 1 <= enrollment[c] <= max_class_size for c in range(len(courses))):
        raise RuntimeError("A course breaks the class-size rules.")
    if any(course_times[c] != t for c, t in locks.course_slots.items()):
        raise RuntimeError("A course does not meet at its required time.")
    if any((c in schedules[s]) != required for (s, c), required in locks.student_courses.items()):
        raise RuntimeError("A student schedule breaks a manual rule.")


# 8. Give staff a short summary and two ways to look up the same schedule.

def write_csv(path, headers, rows):
    # This encoding lets Excel read course and student names correctly.
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def write_outputs(folder, args, courses, students, course_times, schedules, locks, rounds):
    """Write one summary, one row per student, and one row per course."""
    student_count = len(students)
    slot_count = len(SLOT_SIZES)
    total_seats = student_count * slot_count
    rating_name = {"S": "S", "L": "L", "": "Blank"}
    rating_rank = {"S": 2, "L": 1, "": 0}

    # We have the courses each student takes. Build the lists for each course too.
    students_by_course = [[] for _ in courses]
    courses_by_slot = [[] for _ in SLOT_SIZES]
    for c, time in enumerate(course_times):
        courses_by_slot[time].append(c)

    student_rows = []
    ratings_total = Counter()
    best_rated_slots = 0
    students_with_all_preferred = 0
    most_courses_in_one_topic = 0
    total_different_teachers = 0
    for s, schedule in enumerate(schedules):
        student = students[s]
        choices = []
        ratings = Counter()
        best_for_student = 0
        for t, c in enumerate(schedule):
            rating = student.ratings[c]
            choices.append(f"{courses[c].name} [{rating_name[rating]}]")
            students_by_course[c].append(s)
            ratings[rating] += 1

            # This number describes the chosen timetable. Look only at this slot.
            # A forced course sets the choice. Otherwise compare all allowed ratings.
            best_rank = 0
            for other in courses_by_slot[t]:
                rule = locks.student_courses.get((s, other))
                if rule is True:
                    best_rank = rating_rank[student.ratings[other]]
                    break
                if student.ratings[other] != "X" and rule is not False:
                    best_rank = max(best_rank, rating_rank[student.ratings[other]])
            if rating_rank[rating] == best_rank:
                best_for_student += 1

        topics = Counter(courses[c].topic.casefold() for c in schedule if courses[c].topic)
        same_topic = max(topics.values(), default=0)
        different_teachers = len({courses[c].instructor for c in schedule})
        total_different_teachers += different_teachers
        most_courses_in_one_topic = max(most_courses_in_one_topic, same_topic)
        best_rated_slots += best_for_student
        students_with_all_preferred += ratings[""] == 0
        ratings_total.update(ratings)
        student_rows.append([
            student.name, *choices, ratings["S"], ratings["L"], ratings[""],
            best_for_student, different_teachers, sum(topics.values()) - len(topics), same_topic,
        ])

    write_csv(
        folder / "student_schedules.csv",
        ["Student", *(f"Slot {t + 1}" for t in range(slot_count)),
         "Strongly preferred (S)", "Liked (L)", "Acceptable (Blank)",
         "Slots with best available rating", "Different instructors",
         "Topic repeats (known topics)", "Most courses in one known topic"],
        student_rows,
    )

    course_rows = []
    courses_in_ideal_range = 0
    class_sizes = []
    for t, group in enumerate(courses_by_slot):
        # 32 / 3 gives an ideal range of 10 to 11. With 4 courses, it is 8 to 8.
        low = student_count // len(group)
        high = math.ceil(student_count / len(group))
        for c in sorted(group, key=lambda c: courses[c].name):
            course = courses[c]
            enrolled = students_by_course[c]
            genders = Counter(students[s].gender for s in enrolled)
            class_sizes.append(len(enrolled))
            courses_in_ideal_range += low <= len(enrolled) <= high
            names = ", ".join(students[s].name for s in enrolled)
            course_rows.append([
                t + 1, course.name, course.instructor, course.topic, len(enrolled),
                low, high, genders["F"], genders["M"], len(enrolled) - genders["F"] - genders["M"],
                names,
            ])
    write_csv(
        folder / "course_timetable.csv",
        ["Time slot", "Course", "Instructor", "Topic (blank if missing)", "Number of students",
         "Ideal minimum", "Ideal maximum", "Reported F", "Reported M", "Other or missing label",
         "Students in this course"],
        course_rows,
    )

    # Start with counts that can go straight into the paper, then explain the scores.
    known_topics = sum(bool(course.topic) for course in courses)
    proved_rounds = sum(result.proved_through_this_round for result in rounds)
    summary = [
        "MATHILY-ER WEEK OF CHAOS",
        "",
        f"Survey used: {args.survey.name}",
        f"Students: {student_count}   Courses: {len(courses)}   Time slots: {slot_count}",
        f"Check passed: {student_count} different student schedules, one course per slot.",
        "No instructor conflicts, X assignments, or broken staff rules.",
        "",
        "STUDENT PREFERENCES",
    ]
    for rating, label in (("S", "Strongly preferred (S)"), ("L", "Liked (L)"), ("", "Acceptable (blank)")):
        count = ratings_total[rating]
        summary.append(f"{label}: {count} of {total_seats} assignments ({count / total_seats:.1%})")
    summary.extend([
        f"Students with S or L in every slot: {students_with_all_preferred} of {student_count}",
        f"Assignments matching the best rating in their slot: {best_rated_slots} of {total_seats}",
        "That last count compares courses in the chosen slot after X and staff enrollment rules.",
        "",
        "TOPICS, INSTRUCTORS, AND CLASS SIZES",
        f"Courses with a known topic: {known_topics} of {len(courses)}",
        f"Most courses one student took in one known topic: {most_courses_in_one_topic} (limit {MAX_SAME_TOPIC})",
        "Missing topic labels are left out of the topic counts.",
        f"Different instructors per student on average: {total_different_teachers / student_count:.2f}",
        f"Smallest class: {min(class_sizes)}   Largest class: {max(class_sizes)}",
        f"Classes in their ideal size range: {courses_in_ideal_range} of {len(courses)}",
        "The ideal range rounds the average class size in that slot down and up.",
        "",
        "HOW EACH SEARCH ROUND DID",
        f"Consecutive rounds proved best, starting with preferences: {proved_rounds} of {len(rounds)}.",
        f"Preference points: S = {total_seats + 1}, L = 1, blank = 0.",
        "Later rounds keep the scores already reached for earlier goals.",
        "",
    ])
    explanations = [
        "More S choices first. More L choices break ties.",
        "Add the squared F/M gap for each course. A gap of 2 costs 4 points.",
        "Every course after the first in the same known topic costs 1 point per student.",
        "Every course after the first with the same instructor costs 1 point per student.",
        "Each seat above or below a course's ideal size range costs 1 point.",
    ]
    for number, (result, explanation) in enumerate(zip(rounds, explanations), start=1):
        direction = "higher is better" if result.higher_is_better else "lower is better"
        summary.extend([f"{number}. {result.name} ({direction})", f"   {explanation}"])
        if result.proved_through_this_round:
            detail = "Proved best."
        elif result.status == "OPTIMAL":
            detail = "Proved best for the earlier scores kept in this run."
        elif result.best_possible_bound is None:
            detail = "No new schedule found in time. Used the earlier schedule."
        else:
            low, high = sorted((result.score, result.best_possible_bound))
            detail = f"Best possible score is between {low} and {high} with earlier scores fixed."
        summary.extend([f"   Score kept: {result.score}. {detail} Search time: {result.seconds:.1f} s.", ""])
    summary.extend([
        "WHERE TO LOOK NEXT",
        "student_schedules.csv gives one row per student. Each slot shows a course and its survey rating.",
        "course_timetable.csv gives one row per course with its time, counts, and student list.",
        "",
        f"Run settings: {args.seconds_per_stage:g} seconds per round, {args.workers} workers, seed {args.seed}.",
        f"Course times fixed by staff: {len(locks.course_slots)}",
        f"Student choices required or excluded by staff: {len(locks.student_courses)}",
        f"Class-size cap set by staff: {MAX_CLASS_SIZE if MAX_CLASS_SIZE is not None else 'None'}",
    ])
    (folder / "START_HERE.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main():
    """Read the settings, find a schedule, check it, then save it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("survey", type=Path, help="Survey file in TXT, TSV, or CSV format")
    parser.add_argument("--out", type=Path, default=Path("schedule_results"), help="New folder for this run")
    parser.add_argument("--seconds-per-stage", type=float, default=20, help="Search time for each goal")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="Number of solver workers")
    parser.add_argument("--seed", type=int, default=42, help="Number that helps set the search order")
    args = parser.parse_args()
    started = perf_counter()
    try:
        if sorted(SLOT_SIZES) != [3, 3, 4, 4, 4]:
            raise ValueError("Use three four-course slots and two three-course slots.")
        if type(MAX_SAME_TOPIC) is not int or not 1 <= MAX_SAME_TOPIC <= len(SLOT_SIZES):
            raise ValueError("Set MAX_SAME_TOPIC to a whole number from 1 to 5.")
        if not math.isfinite(args.seconds_per_stage) or args.seconds_per_stage <= 0 or args.workers < 1:
            raise ValueError("Use a positive time limit and at least one worker.")
        if not 0 <= args.seed < 2**31:
            raise ValueError("The seed must be between 0 and 2147483647.")
        if args.out.exists():
            raise ValueError(f"{args.out} already exists. Choose a new --out folder.")

        courses, students = read_survey(args.survey)
        if len(students) < max(SLOT_SIZES):
            raise ValueError("There are too few students to put at least one in every class.")
        locks = make_locks(courses, students)
        max_size = len(students) if MAX_CLASS_SIZE is None else MAX_CLASS_SIZE
        if type(max_size) is not int or max_size < 1 or max_size * min(SLOT_SIZES) < len(students):
            raise ValueError("MAX_CLASS_SIZE leaves too few seats in a three-course slot.")

        print(f"Scheduling {len(students)} students and {len(courses)} courses.", flush=True)
        missing_topics = sum(not course.topic for course in courses)
        if missing_topics:
            print(f"{missing_topics} courses have no topic label. Add known topics in TOPIC_OVERRIDES.", flush=True)
        built = build_model(courses, students, SLOT_SIZES, locks, max_size)
        solution, rounds = solve_in_priority_order(
            built, courses, students, SLOT_SIZES, locks, args.seconds_per_stage, args.workers, args.seed,
        )

        # Turn the solver's yes/no answers back into course times and student lists.
        course_times = [solution.value(time) for time in built.course_time]
        schedules = []
        for row in built.takes_course:
            chosen = [c for c, choice in enumerate(row) if solution.value(choice)]
            schedules.append(tuple(sorted(chosen, key=course_times.__getitem__)))
        check_schedule(courses, students, SLOT_SIZES, locks, max_size, course_times, schedules)

        # Write all three files before moving the finished folder into place.
        # A failed write should leave no half-finished results folder.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
            folder = Path(temporary) / "results"
            folder.mkdir()
            write_outputs(folder, args, courses, students, course_times, schedules, locks, rounds)
            folder.rename(args.out)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {len(schedules)} student schedules in {perf_counter() - started:.1f}s.")
    print(f"Open {args.out / 'START_HERE.txt'} first.")


# Run main only when we run this file as a script.
if __name__ == "__main__":
    main()
