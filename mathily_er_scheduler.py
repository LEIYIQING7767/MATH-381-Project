#!/usr/bin/env python3
"""Schedule the MathILy-Er Week of Chaos with OR-Tools CP-SAT.

The camp supplies 18 already chosen courses and the students' survey ratings.
The script decides when each course meets and which course each student takes
in each of the five slots. These decisions are linked because two courses
placed in the same slot compete for the same students.

Running the script

    python -m pip install ortools==9.15.6755
    python mathily_er_scheduler.py "Pasted text (2).txt" --out results_2

The survey has one column per course. Its first three rows give the instructor,
class name, and topic. Each following row gives one student's ratings. S means
strongly preferred, L means liked, blank means acceptable, and X excludes the
course. A student label such as "Student 1 (F)" supplies the reported F/M label.

How to read the code

1. read_survey reads the course and student records. make_locks translates any
   manual rules near the top of the file into course and student indices.
2. build_model creates the choices and the rules that every schedule must obey.
3. solve_in_priority_order asks the solver to improve preferences, gender
   balance, topic variety, instructor variety, and class sizes in that order.
   prune_timetables can speed up the later rounds using the fixed preference score.
4. validate_schedule checks the chosen timetable and enrollments. write_outputs
   turns them into CSV tables. main calls these functions in order.

Changing a schedule

Uncomment or add a rule in STUDENT_COURSE_RULES. True requires a course and
False excludes it. Use "Student 2" without the survey's "(F)" or "(M)" suffix.
If the course must also meet at a particular time, set COURSE_SLOT_RULES.
Rerun with a fresh output folder, such as --out results_adjusted.
MAX_SAME_TOPIC limits each student's count in a topic to 3 by default.
Change it to 4 to allow four. Add confirmed tags in TOPIC_OVERRIDES to include
courses with a blank topic cell in the topic rules and scores.

Reading a run

student_schedules.csv shows one row per student. timetable.csv shows one row
per course, and rosters.csv shows one row per enrollment. results_table.csv
summarizes the schedule. objective_scores.csv shows which priorities were
proved optimal and the remaining bounds. The preference stage counts S and L
across the cohort, and the later stages preserve the counts reached.

OR-Tools reference
https://developers.google.com/optimization/cp/cp_solver
School scheduling example
https://github.com/google/or-tools/blob/stable/examples/contrib/school_scheduling_sat.py
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

# Edit these settings to change the requirements for a run.
# Course titles must match the survey, and slot numbers here start at 1.
MAX_SAME_TOPIC = 3                       # Hard limit per student for each known topic.
MAX_CLASS_SIZE = None                    # None gives each course up to the cohort size.
SLOT_SIZES = (4, 4, 4, 3, 3)
TOPIC_OVERRIDES = {
    # "Exact course title": "Topic tag",  # Use the same tag for courses in the same topic.
}
STUDENT_COURSE_RULES = [
    # ("Student 2", "Algebraists Anonymous", True),    # Require this course.
    # ("Student 2", "Complaints in Noir Films", False),  # Exclude this course.
]
COURSE_SLOT_RULES = {
    # "Algebraists Anonymous": 2,        # Keep this course in slot 2.
}


@dataclass(frozen=True)
class Course:
    name: str
    instructor: str
    topic: str


@dataclass(frozen=True)
class Student:
    name: str
    gender: str
    ratings: tuple[str, ...]


@dataclass
class Locks:
    course_slots: dict[int, int]
    student_courses: dict[tuple[int, int], bool]


@dataclass
class ScheduleModel:
    """The solver's model and references to the variables we read after solving."""

    model: cp_model.CpModel
    y: list[list[cp_model.IntVar]]
    timetable: list[cp_model.IntVar]
    s_count: cp_model.LinearExpr | int
    l_count: cp_model.LinearExpr | int
    objectives: list[tuple[str, cp_model.LinearExpr | int, bool]]
    weights: dict[str, int]


def read_survey(path):
    """Turn the survey table into Course and Student records in column and row order."""
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(source, delimiter=delimiter)
            if any(cell.strip() for cell in row)
        ]
    if len(rows) < 4 or [row[0].casefold() for row in rows[:3]] != [
        "instructor", "class name", "topic"
    ]:
        raise ValueError("Expected Instructor, Class name and Topic in the first three rows.")

    names = rows[1][1:]
    while names and not names[-1]:
        names.pop()
    if len(names) != 18 or not all(names) or len(set(names)) != 18:
        raise ValueError("Expected 18 distinct, consecutive class names.")

    def cells(row):
        # Drop the label column and pad any trailing blank cells omitted by the export.
        if any(row[19:]):
            raise ValueError(f"Unexpected values beyond the last course in row {row[0]!r}.")
        return (row[1:] + [""] * 18)[:18]

    instructors, topics = cells(rows[0]), cells(rows[2])
    if not all(instructors):
        raise ValueError("Every course needs an instructor.")
    unknown_topics = TOPIC_OVERRIDES.keys() - set(names)
    if unknown_topics:
        raise ValueError(f"Unknown course in TOPIC_OVERRIDES: {', '.join(sorted(unknown_topics))}.")
    topics = [TOPIC_OVERRIDES.get(name, topic).strip() for name, topic in zip(names, topics)]
    courses = [Course(*values) for values in zip(names, instructors, topics)]
    students = []
    seen = set()
    for row in rows[3:]:
        match = re.fullmatch(r"(.*?)\s*\(([^()]*)\)", row[0])
        name, gender = (match[1].strip(), match[2].strip().upper()) if match else (row[0], "")
        if gender == "W":
            gender = "F"
        if not name or name in seen:
            raise ValueError(f"Missing or duplicate student ID in row {row[0]!r}.")
        seen.add(name)
        ratings = tuple(re.sub(r"[\ufe0e\ufe0f]", "", cell).upper() for cell in cells(row))
        for course, rating in zip(courses, ratings):
            if rating not in ("S", "L", "", "X"):
                raise ValueError(f"Invalid rating {rating!r} for {name}, {course.name!r}.")
        students.append(Student(name, gender, ratings))
    return courses, students


def make_locks(courses, students, slot_count):
    """Convert manual rules written with names into solver constraints by index."""
    course_ids = {course.name: c for c, course in enumerate(courses)}
    student_ids = {student.name: s for s, student in enumerate(students)}
    locks = Locks({}, {})
    for name, slot in COURSE_SLOT_RULES.items():
        if name not in course_ids:
            raise ValueError(f"Unknown course in COURSE_SLOT_RULES: {name!r}.")
        if type(slot) is not int or not 1 <= slot <= slot_count:
            raise ValueError(f"Slot for {name!r} must be an integer from 1 to {slot_count}.")
        locks.course_slots[course_ids[name]] = slot - 1  # Staff slots start at 1, Python indices at 0.
    for name, course, required in STUDENT_COURSE_RULES:
        if name not in student_ids or course not in course_ids or type(required) is not bool:
            raise ValueError(f"Expected a known student, a known course and True/False for {name}, {course!r}.")
        s, c = student_ids[name], course_ids[course]
        if required and students[s].ratings[c] == "X":
            raise ValueError(f"{name} cannot be required to take X-rated course {course!r}.")
        if (s, c) in locks.student_courses and locks.student_courses[s, c] != required:
            raise ValueError(f"Conflicting rules for {name} and {course!r}.")
        locks.student_courses[s, c] = required
    return locks


def objective_weights(n_students, n_slots, n_f, n_m):
    """Compute the weights for an equivalent single score, listed high to low.

    Start with class size at weight 1. A safe bound for its total penalty is
    320 in a 32-student cohort, so a repeated instructor gets weight 321.
    At each step, multiply by one plus the lower goal's bound. The resulting
    weight makes a one-unit improvement beat every possible lower-priority
    change combined. The solver uses separate rounds to apply this order.
    """
    seats = n_students * n_slots
    # A goal's weight exceeds the maximum weighted swing of all lower goals.
    # These ranges are safe upper bounds, even when some tags are missing.
    bounds = {
        "S": seats,
        "L": seats,
        "gender_penalty": n_slots * (n_f**2 + n_m**2),
        "topic_repeats": n_students * (n_slots - 1),
        "instructor_repeats": n_students * (n_slots - 1),
        "size_penalty": 2 * seats,
    }
    weights = {}
    next_weight = 1
    for name, bound in reversed(list(bounds.items())):
        weights[name] = next_weight
        next_weight *= bound + 1
    return {name: weights[name] for name in bounds}


def build_model(courses, students, slot_sizes, locks, max_class_size):
    """Create the solver's choices, mandatory rules, and scoring expressions."""
    model = cp_model.CpModel()
    course_ids = range(len(courses))
    student_ids = range(len(students))
    slots = range(len(slot_sizes))
    n_students = len(students)
    instructors = defaultdict(list)
    topics = defaultdict(list)
    for c, course in enumerate(courses):
        instructors[course.instructor].append(c)
        if course.topic:
            topics[course.topic.casefold()].append(c)

    # A Boolean choice has value 0 or 1. x[c][t] is 1 when course c meets in
    # slot t. y[s][c] is 1 when student s takes course c.
    x = [[model.new_bool_var(f"x_{c}_{t}") for t in slots] for c in course_ids]
    y = [[model.new_bool_var(f"y_{s}_{c}") for c in course_ids] for s in student_ids]
    timetable = [model.new_int_var(0, len(slots) - 1, f"slot_of_course_{c}") for c in course_ids]
    # Each course meets in exactly one slot. Count the courses and check
    # instructors within each slot, then enforce any manually fixed times.
    for c in course_ids:
        model.add_exactly_one(x[c])
        model.add(timetable[c] == sum(t * x[c][t] for t in slots))
    for t in slots:
        model.add(sum(x[c][t] for c in course_ids) == slot_sizes[t])
        for group in instructors.values():
            model.add_at_most_one(x[c][t] for c in group)
    for c, t in locks.course_slots.items():
        model.add(x[c][t] == 1)

    # Exchanging two equally sized slots with no fixed course times gives
    # the same scores. Order these slots by their first course to search
    # only one representative of each equivalent arrangement.
    for size in set(slot_sizes):
        free_slots = [t for t in slots if slot_sizes[t] == size and t not in locks.course_slots.values()]
        first_courses = []
        for t in free_slots:
            first = model.new_int_var(0, len(courses) - 1, f"first_course_{t}")
            model.add_min_equality(first, [c + len(courses) * (1 - x[c][t]) for c in course_ids])
            first_courses.append(first)
        for earlier, later in zip(first_courses, first_courses[1:]):
            model.add(earlier < later)

    # z[s,c,t] is 1 when the student takes that course in that slot.
    # z <= x lets a student choose a course only when it meets in that slot.
    # The sum of z over slots gives y. The sum of z over courses in a slot
    # must be 1 for each student.
    signatures = []
    for s, student in enumerate(students):
        choices_by_slot = [[] for _ in slots]
        for c, rating in enumerate(student.ratings):
            lock = locks.student_courses.get((s, c))
            if rating == "X" or lock is False:
                model.add(y[s][c] == 0)
                continue
            if lock is True:
                model.add(y[s][c] == 1)
            choices_for_course = []
            for t in ([locks.course_slots[c]] if c in locks.course_slots else slots):
                z = model.new_bool_var(f"z_{s}_{c}_{t}")
                model.add(z <= x[c][t])
                choices_for_course.append(z)
                choices_by_slot[t].append(z)
            model.add(sum(choices_for_course) == y[s][c])
        for choices in choices_by_slot:
            model.add_exactly_one(choices)
        model.add(sum(y[s]) == len(slots))

        # Give each course its own power of two (1, 2, 4, 8, ...).
        # Adding the selected courses' numbers identifies the exact set.
        # For example, choosing courses with values 1 and 4 gives code 5.
        signature = model.new_int_var(0, (1 << len(courses)) - 1, f"schedule_{s}")
        model.add(signature == sum((1 << c) * y[s][c] for c in course_ids))
        signatures.append(signature)
    model.add_all_different(signatures)

    # Each selected course with the matching rating adds 1 to its rating count.
    s_count = sum(y[s][c] for s in student_ids for c in course_ids if students[s].ratings[c] == "S")
    l_count = sum(y[s][c] for s in student_ids for c in course_ids if students[s].ratings[c] == "L")
    group_f = [s for s in student_ids if students[s].gender == "F"]
    group_m = [s for s in student_ids if students[s].gender == "M"]
    weights = objective_weights(n_students, len(slots), len(group_f), len(group_m))
    # One S is worth total seats + 1 points. With 32 students and 5 slots,
    # that gives 161 points for S and 1 for L in the preference round.
    satisfaction = (weights["S"] // weights["L"]) * s_count + l_count
    gender_penalties, size_penalties = [], []
    for c in course_ids:
        enrollment = sum(y[s][c] for s in student_ids)
        model.add_linear_constraint(enrollment, 1, max_class_size)

        # Square the difference between reported M and F counts in this course.
        # A gap of 3 contributes 9 points. A gap of 1 contributes 1 point.
        imbalance = model.new_int_var(0, max(len(group_f), len(group_m)), f"imbalance_{c}")
        model.add_abs_equality(
            imbalance, sum(y[s][c] for s in group_m) - sum(y[s][c] for s in group_f)
        )
        squared = model.new_int_var(0, max(len(group_f), len(group_m)) ** 2, f"gender_penalty_{c}")
        model.add_element(imbalance, [k * k for k in range(max(len(group_f), len(group_m)) + 1)], squared)
        gender_penalties.append(squared)

        # Compare enrollment to an even share of the students in this slot.
        # With 32 students and 3 courses, 10 or 11 students is ideal.
        # A class of 14 students in that slot contributes a penalty of 3.
        target_min = sum((n_students // slot_sizes[t]) * x[c][t] for t in slots)
        target_max = sum(math.ceil(n_students / slot_sizes[t]) * x[c][t] for t in slots)
        size_penalty = model.new_int_var(0, n_students, f"size_penalty_{c}")
        model.add_max_equality(size_penalty, [0, target_min - enrollment, enrollment - target_max])
        size_penalties.append(size_penalty)

    def repeated_groups(groups, name, cap=None):
        """Count courses after the first in each topic or instructor group."""
        repeats = []
        for s in student_ids:
            for g, group in enumerate(groups.values()):
                if len(group) > 1:
                    count = sum(y[s][c] for c in group)
                    # With three algebra courses the repeat penalty is 2.
                    # The topic cap also makes a fourth algebra course invalid at cap 3.
                    if cap is not None and len(group) > cap:
                        model.add(count <= cap)
                    repeat = model.new_int_var(0, len(group) - 1, f"{name}_repeat_{s}_{g}")
                    model.add_max_equality(repeat, [0, count - 1])
                    repeats.append(repeat)
        return sum(repeats)

    # For each slot, splitting the signed F/M difference as evenly as possible
    # across its courses gives a lower bound on its sum of squared differences.
    gender_lower_bound = 0
    for size in slot_sizes:
        quotient, remainder = divmod(len(group_m) - len(group_f), size)
        gender_lower_bound += (size - remainder) * quotient ** 2 + remainder * (quotient + 1) ** 2
    model.add(sum(gender_penalties) >= gender_lower_bound)

    # Each tuple holds a stage name, its score, and whether higher is better.
    objectives = [
        ("satisfaction", satisfaction, True),
        ("gender_penalty", sum(gender_penalties), False),
        ("topic_repeats", repeated_groups(topics, "topic", MAX_SAME_TOPIC), False),
        ("instructor_repeats", repeated_groups(instructors, "instructor"), False),
        ("size_penalty", sum(size_penalties), False),
    ]
    return ScheduleModel(model, y, timetable, s_count, l_count, objectives, weights)


def prune_timetables(built, courses, students, slot_sizes, target, limit=200):
    """Screen possible timetables using the fixed preference score as a cutoff.

    Give every student the best rating they could get in each slot based on the
    survey alone. Their total is an optimistic upper bound. Even that optimistic
    result has to reach the fixed score for the timetable to be worth searching.
    If many timetables pass this test, the solver uses its ordinary search.
    """
    n_courses = len(courses)
    points = {"S": len(students) * len(slot_sizes) + 1, "L": 1, "": 0, "X": -1}
    ratings = [[points[rating] for rating in student.ratings] for student in students]
    # A block is a set of courses offered together in one slot.
    blocks_by_first = [[] for _ in courses]
    for size in set(slot_sizes):
        for block in combinations(range(n_courses), size):
            if len({courses[c].instructor for c in block}) < size:
                continue
            best_ratings = [max(row[c] for c in block) for row in ratings]
            if min(best_ratings) >= 0:
                mask = sum(1 << c for c in block)  # One bit per course present in the block.
                blocks_by_first[block[0]].append((mask, block, sum(best_ratings)))

    def next_blocks(mask, threes, fours):
        first = (mask & -mask).bit_length() - 1
        for bits, block, score in blocks_by_first[first]:
            n3, n4 = threes - (len(block) == 3), fours - (len(block) == 4)
            if n3 >= 0 and n4 >= 0 and mask & bits == bits:
                yield mask ^ bits, n3, n4, block, score

    @cache
    def best_remaining(mask, threes, fours):
        # Remember the best upper bound for the same set of remaining courses
        # and the same numbers of three- and four-course slots.
        if mask == 0:
            return 0
        return max(
            (score + best_remaining(rest, n3, n4) for rest, n3, n4, _, score in next_blocks(mask, threes, fours)),
            default=-math.inf,
        )

    patterns = []

    def collect(mask, threes, fours, needed, blocks):
        if not mask:
            blocks_by_size = {size: iter([block for block in blocks if len(block) == size]) for size in set(slot_sizes)}
            pattern = [0] * n_courses
            for t, size in enumerate(slot_sizes):
                for c in next(blocks_by_size[size]):
                    pattern[c] = t
            patterns.append(tuple(pattern))
            return
        for rest, n3, n4, block, score in next_blocks(mask, threes, fours):
            if score + best_remaining(rest, n3, n4) >= needed:
                collect(rest, n3, n4, needed - score, [*blocks, block])
                if len(patterns) > limit:
                    return

    collect((1 << n_courses) - 1, slot_sizes.count(3), slot_sizes.count(4), target, [])
    if len(patterns) > limit:
        return
    if not patterns:
        raise RuntimeError("The timetable bound excluded a previously feasible schedule.")
    built.model.add_allowed_assignments(built.timetable, patterns)

    # A single low-rated assignment cannot lose more points than the total
    # difference between a timetable's upper bound and the locked score.
    viable = [[False] * n_courses for _ in students]
    for pattern in patterns:
        by_slot = [[c for c, t in enumerate(pattern) if t == slot] for slot in range(len(slot_sizes))]
        best = [[max(row[c] for c in block) for block in by_slot] for row in ratings]
        slack = sum(map(sum, best)) - target
        for s, row in enumerate(ratings):
            for c, rating in enumerate(row):
                if rating >= 0 and best[s][pattern[c]] - rating <= slack:
                    viable[s][c] = True
    for s, allowed in enumerate(viable):
        for c, can_take in enumerate(allowed):
            if not can_take:
                built.model.add(built.y[s][c] == 0)
    print(f"Timetables within the preference bound: {len(patterns)}", flush=True)


def solve_in_priority_order(built, courses, students, slot_sizes, locks, seconds, workers, seed):
    """Solve one priority at a time, then require its score in later rounds.

    OPTIMAL means the best score was proved for the current model. FEASIBLE
    means the solver found a legal schedule and a bound on how much better
    the score might be. The best score found is fixed in either case.
    """
    model = built.model
    best = None
    stages = []
    earlier_proven = True
    variables = [model.get_int_var_from_proto_index(i) for i in range(len(model.proto.variables))]
    for name, objective, maximize in built.objectives:
        if maximize:
            model.maximize(objective)
        else:
            model.minimize(objective)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds
        solver.parameters.num_workers = workers
        solver.parameters.random_seed = seed
        status = solver.solve(model)
        if status == cp_model.MODEL_INVALID:
            raise RuntimeError(f"Invalid optimization model: {model.validate() or solver.solution_info()}")
        if best is None and status == cp_model.INFEASIBLE:
            raise ValueError("No feasible schedule exists with these inputs and locks.")
        if best is None and status == cp_model.UNKNOWN:
            raise ValueError("No feasible schedule found within the time limit. Increase --seconds-per-stage.")
        if status == cp_model.INFEASIBLE:
            raise RuntimeError("A previously feasible score became infeasible.")
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            bound = round(solver.best_objective_bound)
            sign = 1 if maximize else -1
            if best is None or sign * solver.value(objective) >= sign * best.value(objective):
                best = solver
        else:
            bound = None
        value = best.value(objective)
        # A later stage can be optimal for the scores fixed in earlier stages.
        # Overall proof through that stage also requires proof of those scores.
        proven = status == cp_model.OPTIMAL
        stages.append({
            "objective": name,
            "direction": "maximize" if maximize else "minimize",
            "value": value,
            "bound_at_fixed_higher_scores": bound,
            "status": solver.status_name(status),
            "proved_at_all_higher_optima": proven and earlier_proven,
            "seconds": round(solver.wall_time, 3),
        })
        earlier_proven = earlier_proven and proven
        print(f"{name}: {value} ({solver.status_name(status)}, bound {bound})", flush=True)
        if name == "satisfaction":
            model.add(built.s_count == best.value(built.s_count))
            model.add(built.l_count == best.value(built.l_count))
            if not locks.course_slots:
                prune_timetables(built, courses, students, slot_sizes, value)
        else:
            model.add(objective == value)
        # Give the next round the previous legal schedule as a starting hint.
        model.clear_hints()
        for var in variables:
            model.add_hint(var, best.value(var))
    return best, stages


def validate_schedule(courses, students, slot_sizes, locks, max_class_size, course_slots, schedules):
    """Check the exported solution independently of the solver's constraints."""
    if Counter(course_slots) != Counter(dict(enumerate(slot_sizes))):
        raise RuntimeError("The timetable has incorrect numbers of courses per slot.")
    for t in range(len(slot_sizes)):
        instructors = [courses[c].instructor for c, slot in enumerate(course_slots) if slot == t]
        if len(instructors) != len(set(instructors)):
            raise RuntimeError("The timetable double-books an instructor.")
    if len(schedules) != len(students) or len(set(schedules)) != len(students):
        raise RuntimeError("The solution has missing or duplicate student schedules.")
    enrollment = Counter()
    for s, schedule in enumerate(schedules):
        if len(schedule) != len(slot_sizes) or any(course_slots[c] != t for t, c in enumerate(schedule)):
            raise RuntimeError("A student does not have exactly one course in each slot.")
        if any(students[s].ratings[c] == "X" for c in schedule):
            raise RuntimeError("A student was assigned an X-rated course.")
        topics = Counter(courses[c].topic.casefold() for c in schedule if courses[c].topic)
        if any(count > MAX_SAME_TOPIC for count in topics.values()):
            raise RuntimeError("A student exceeds the topic cap.")
        enrollment.update(schedule)
    if any(not 1 <= enrollment[c] <= max_class_size for c in range(len(courses))):
        raise RuntimeError("A course violates the class-size limits.")
    if any(course_slots[c] != t for c, t in locks.course_slots.items()):
        raise RuntimeError("The timetable violates a locked slot.")
    if any((c in schedules[s]) != required for (s, c), required in locks.student_courses.items()):
        raise RuntimeError("The schedule violates a locked enrollment.")


def write_outputs(directory, courses, students, slot_sizes, course_slots, schedules, locks, stages, weights):
    """Write the course, enrollment, student, solver, and summary CSV tables."""
    def write_csv(filename, headers, rows):
        with (directory / filename).open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(headers)
            writer.writerows(rows)

    # The solver returns courses by student. Reverse that mapping for rosters.
    n_students = len(students)
    rosters = [[] for _ in courses]
    courses_by_slot = [[] for _ in slot_sizes]
    for c, t in enumerate(course_slots):
        courses_by_slot[t].append(c)
    for s, schedule in enumerate(schedules):
        for c in schedule:
            rosters[c].append(s)

    class_rows, roster_rows = [], []
    for c in sorted(range(len(courses)), key=lambda c: (course_slots[c], courses[c].name)):
        course, t = courses[c], course_slots[c]
        genders = Counter(students[s].gender for s in rosters[c])
        ratings = Counter(students[s].ratings[c] for s in rosters[c])
        enrollment = len(rosters[c])
        low, high = n_students // slot_sizes[t], math.ceil(n_students / slot_sizes[t])
        class_rows.append([
            t + 1, course.name, course.instructor, course.topic, enrollment, genders["F"], genders["M"],
            enrollment - genders["F"] - genders["M"], ratings["S"], ratings["L"], ratings[""],
            low, high, (genders["M"] - genders["F"]) ** 2, max(0, low - enrollment, enrollment - high),
        ])
        roster_rows.extend(
            [t + 1, course.name, students[s].name, students[s].ratings[c] or "Blank"] for s in rosters[c]
        )
    write_csv(
        "timetable.csv",
        ["Slot", "Course", "Instructor", "Topic", "Enrollment", "F", "M", "Other or unlabeled",
         "S", "L", "Blank", "Ideal min", "Ideal max", "Gender penalty", "Size penalty"],
        class_rows,
    )
    write_csv("rosters.csv", ["Slot", "Course", "Student", "Survey rating"], roster_rows)

    student_rows = []
    counts = Counter()
    best_slots, distinct_instructors, largest_topic_count = 0, 0, 0
    rank = {"S": 2, "L": 1, "": 0}
    for s, (student, schedule) in enumerate(zip(students, schedules)):
        ratings = Counter(student.ratings[c] for c in schedule)
        counts.update(ratings)
        topics = Counter(courses[c].topic.casefold() for c in schedule if courses[c].topic)
        max_topic_count = max(topics.values(), default=0)
        largest_topic_count = max(largest_topic_count, max_topic_count)
        instructor_count = len({courses[c].instructor for c in schedule})
        distinct_instructors += instructor_count
        # Compare the assigned rating with the best survey rating in that slot
        # after X exclusions and manual enrollment rules. This is a report metric.
        best_for_student = 0
        for t, c in enumerate(schedule):
            eligible = [
                d for d in courses_by_slot[t] if student.ratings[d] != "X" and locks.student_courses.get((s, d)) is not False
            ]
            required = [d for d in eligible if locks.student_courses.get((s, d)) is True]
            best_rating = max(rank[student.ratings[d]] for d in required or eligible)
            best_for_student += rank[student.ratings[c]] == best_rating
        best_slots += best_for_student
        student_rows.append([
            student.name, *(courses[c].name for c in schedule),
            ratings["S"], ratings["L"], ratings[""], best_for_student,
            instructor_count, sum(topics.values()) - len(topics), max_topic_count,
        ])
    write_csv(
        "student_schedules.csv",
        ["Student", *(f"Slot {t + 1}" for t in range(len(slot_sizes))),
         "S", "L", "Blank", "Best rated slots", "Distinct instructors", "Topic repeats", "Largest topic count"],
        student_rows,
    )
    write_csv(
        "objective_scores.csv", [*stages[0], "weight in equivalent objective"],
        [
            [*stage.values(),
             f"S={weights['S']}, L={weights['L']}" if stage["objective"] == "satisfaction" else -weights[stage["objective"]]]
            for stage in stages
        ],
    )

    score = {stage["objective"]: stage["value"] for stage in stages}
    tagged_courses = sum(bool(course.topic) for course in courses)
    seat_count = len(slot_sizes) * n_students
    ideal_size_count = sum(row[-1] == 0 for row in class_rows)
    proven = sum(stage["proved_at_all_higher_optima"] for stage in stages)

    def fraction(count, total):
        return f"{count}/{total} ({count / total:.1%})"

    write_csv("results_table.csv", ["Measure", "Result", "How to read it"], [
        ("Cohort size", n_students, "Each student takes one course in each of the five slots."),
        ("S assignments", fraction(counts["S"], seat_count), "Share of seats rated strongly preferred by their assigned student."),
        ("L assignments", fraction(counts["L"], seat_count), "Share of seats rated liked by their assigned student."),
        ("Blank assignments", fraction(counts[""], seat_count), "Share of seats with a neutral survey rating."),
        ("Best rating in slot", fraction(best_slots, seat_count), "Assigned rating matched the best rating in that slot after exclusions and locks."),
        ("Gender imbalance", score["gender_penalty"], "Sum of squared differences between M and F counts across classes. Lower is better."),
        ("Courses with topic tags", fraction(tagged_courses, len(courses)), "Topic statistics cover the supplied and overridden tags."),
        ("Topic repeats", score["topic_repeats"], "Each tagged course after the first in the same topic adds one repeat."),
        ("Largest same-topic count", largest_topic_count, f"Largest tagged-topic count for one student. The cap is {MAX_SAME_TOPIC}."),
        ("Mean distinct instructors", f"{distinct_instructors / n_students:.2f}", "Average per student, out of five."),
        ("Unique student schedules", fraction(len(set(schedules)), n_students), "Every student has a distinct set of five courses."),
        ("Classes in ideal size band", fraction(ideal_size_count, len(courses)), "Ideal is the floor or ceiling of cohort size divided by courses in the slot."),
        ("Class size range", f"{min(map(len, rosters))}–{max(map(len, rosters))}", "Smallest and largest classes across all five slots."),
        ("Class size penalty", score["size_penalty"], "Total seats above or below the ideal bands. Lower is better."),
        ("Priority tiers proven optimal", f"{proven}/{len(stages)}", "Proof includes all higher tiers. See objective_scores.csv for bounds."),
    ])


def main():
    """Run the full workflow from command-line arguments to validated CSVs."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog='Example: python mathily_er_scheduler.py "Pasted text (2).txt" --out results_2',
    )
    parser.add_argument("survey", type=Path, help="Pasted tab-separated survey (.txt or .tsv), or a CSV export")
    parser.add_argument("--out", type=Path, default=Path("schedule_results"), help="Folder for this run's results")
    parser.add_argument("--seconds-per-stage", type=float, default=20, help="Solver time limit per priority")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = perf_counter()
    try:
        if sorted(SLOT_SIZES) != [3, 3, 4, 4, 4]:
            raise ValueError("Use three four-course slots and two three-course slots.")
        if type(MAX_SAME_TOPIC) is not int or not 1 <= MAX_SAME_TOPIC <= len(SLOT_SIZES):
            raise ValueError("MAX_SAME_TOPIC must be an integer from 1 to 5.")
        if not math.isfinite(args.seconds_per_stage) or args.seconds_per_stage <= 0 or args.workers < 1:
            raise ValueError("Use a positive time limit and at least one worker.")
        if not 0 <= args.seed < 2**31:
            raise ValueError("The seed must be between 0 and 2147483647.")
        if args.out.exists():
            raise ValueError(f"{args.out} already exists. Choose a new --out folder for this run.")
        courses, students = read_survey(args.survey)
        if len(students) < max(SLOT_SIZES):
            raise ValueError("There are too few students to run every class with at least one student.")
        locks = make_locks(courses, students, len(SLOT_SIZES))
        cap = len(students) if MAX_CLASS_SIZE is None else MAX_CLASS_SIZE
        if type(cap) is not int or cap < 1 or cap * min(SLOT_SIZES) < len(students):
            raise ValueError("The class-size cap leaves too few seats in a three-course slot.")
        missing_topics = sum(not course.topic for course in courses)
        if missing_topics:
            print(f"{missing_topics} courses lack a topic tag. Add confirmed tags in TOPIC_OVERRIDES to include them.")
        built = build_model(courses, students, SLOT_SIZES, locks, cap)
        signed_weights = {name: weight if name in ("S", "L") else -weight for name, weight in built.weights.items()}
        print("Equivalent objective weights: " + ", ".join(f"{name}={weight:+d}" for name, weight in signed_weights.items()), flush=True)
        solution, stages = solve_in_priority_order(
            built, courses, students, SLOT_SIZES, locks, args.seconds_per_stage, args.workers, args.seed
        )
        # Convert solver variables into slot numbers and student course lists.
        course_slots = [solution.value(slot) for slot in built.timetable]
        schedules = [
            tuple(sorted((c for c, var in enumerate(row) if solution.value(var)), key=course_slots.__getitem__))
            for row in built.y
        ]
        validate_schedule(courses, students, SLOT_SIZES, locks, cap, course_slots, schedules)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
            results = Path(temporary) / "results"
            results.mkdir()
            write_outputs(results, courses, students, SLOT_SIZES, course_slots, schedules, locks, stages, built.weights)
            results.rename(args.out)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {len(schedules)} unique schedules to {args.out} in {perf_counter() - started:.1f}s.")


if __name__ == "__main__":
    main()
