"""Seeded synthetic departments shaped like the proposal's scope (Section 1).

Defaults: 30 faculty, 10 student groups x 6 courses = 60 course sections,
15 lecture rooms + 5 labs, a 5-day week of 8 slots with a lunch slot.
"""

from __future__ import annotations

import random
from collections import Counter

from .instance import (
    Calendar,
    Faculty,
    Group,
    Instance,
    Room,
    RoomType,
    Session,
    SessionKind,
)
from .schemas import Role

# Lab equipment. Routers exist in exactly one lab (L-2), so generated sessions
# never need them; scenarios use them to plant resource contention.
LAB_EQUIPMENT = [["gpu"], ["gpu", "routers"], ["electronics"], [], ["electronics", "gpu"]]
PRACTICAL_NEEDS = [[], ["gpu"], ["electronics"]]

SURNAMES = [
    "Rao", "Menon", "Iyer", "Khan", "Das", "Pillai", "Sharma", "Gupta", "Nair", "Reddy",
    "Banerjee", "Mukherjee", "Joshi", "Kulkarni", "Patel", "Shah", "Singh", "Verma",
    "Chatterjee", "Bose", "Kapoor", "Mehta", "Desai", "Naidu", "Hegde", "Kumar",
    "Agarwal", "Saxena", "Mishra", "Tiwari",
]
COURSE_TITLES = [
    "Data Structures", "Operating Systems", "Computer Networks", "Database Systems",
    "Machine Learning", "Compiler Design", "Theory of Computation", "Computer Architecture",
    "Software Engineering", "Discrete Mathematics", "Digital Logic", "Algorithms",
    "Computer Graphics", "Information Security", "Distributed Systems", "Artificial Intelligence",
    "Probability and Statistics", "Linear Algebra", "Signals and Systems", "Cloud Computing",
]


def faculty_name(i: int) -> str:
    base = SURNAMES[i % len(SURNAMES)]
    return f"Dr. {base}" if i < len(SURNAMES) else f"Dr. {base} {i // len(SURNAMES) + 1}"


def generate_department(
    seed: int = 0,
    *,
    n_faculty: int = 30,
    n_groups: int = 10,
    courses_per_group: int = 6,
    lectures_per_course: int = 3,
    lab_share: float = 0.4,
    n_lecture_rooms: int = 15,
    n_labs: int = 5,
    calendar: Calendar | None = None,
    shared_courses: bool = False,
    co_teaching: float = 0.0,
) -> Instance:
    """``shared_courses``: courses as departments usually have them: each course (one ID, one name) is taken by
    every section, with the same teacher(s) for all of them; ``courses_per_group`` is then the number of courses.
    ``co_teaching``: the share of those courses with two teachers sharing the classes. Both off by default: the
    departments made so far (the app's, the tests', the solver's) are unchanged."""
    if shared_courses:
        return _shared(seed, n_faculty, n_groups, courses_per_group, lectures_per_course, lab_share,
                       n_lecture_rooms, n_labs, calendar, co_teaching)
    rng = random.Random(seed)

    rooms = [
        Room(id=f"R-{101 + i}", name=f"Room {101 + i}", capacity=rng.choice([60, 60, 80, 120]), type=RoomType.LECTURE)
        for i in range(n_lecture_rooms)
    ] + [
        Room(
            id=f"L-{i + 1}",
            name=f"Lab {i + 1}",
            capacity=60,
            type=RoomType.LAB,
            equipment=LAB_EQUIPMENT[i % len(LAB_EQUIPMENT)],
        )
        for i in range(n_labs)
    ]

    hod = "F-101"
    faculty = [
        Faculty(
            id=f"F-{101 + i}",
            name=faculty_name(i),
            department="CSE",
            role=Role.HOD if i == 0 else Role.FACULTY,
            reports_to=None if i == 0 else hod,
        )
        for i in range(n_faculty)
    ]
    groups = [Group(id=f"G-{i + 1:02d}", name=f"Section {i + 1}", size=rng.randint(40, 60)) for i in range(n_groups)]

    load: Counter[str] = Counter()
    sessions: list[Session] = []
    titles: dict[str, str] = {}
    for gi, g in enumerate(groups):
        for k in range(courses_per_group):
            course = f"C-{g.id[2:]}{k + 1}"
            titles[course] = COURSE_TITLES[(gi * courses_per_group + k) % len(COURSE_TITLES)]
            teacher = min(faculty, key=lambda f: (load[f.id], rng.random())).id
            for j in range(lectures_per_course):
                sessions.append(
                    Session(
                        id=f"{course}-L{j + 1}",
                        course=course,
                        kind=SessionKind.LECTURE,
                        faculty=teacher,
                        groups=[g.id],
                    )
                )
            load[teacher] += lectures_per_course
            if rng.random() < lab_share:
                sessions.append(
                    Session(
                        id=f"{course}-P",
                        course=course,
                        kind=SessionKind.PRACTICAL,
                        faculty=teacher,
                        groups=[g.id],
                        duration=2,
                        room_type=RoomType.LAB,
                        equipment=rng.choice(PRACTICAL_NEEDS),
                    )
                )
                load[teacher] += 2

    return Instance(
        name=f"synthetic-cse-s{seed}",
        calendar=calendar or Calendar(),
        rooms=rooms,
        faculty=faculty,
        groups=groups,
        sessions=sessions,
        course_titles=titles,
    )


def _shared(seed: int, n_faculty: int, n_groups: int, n_courses: int, lectures_per_course: int, lab_share: float,
            n_lecture_rooms: int, n_labs: int, calendar: Calendar | None, co_teaching: float) -> Instance:
    """``generate_department`` with shared courses: one course for every section, the same teacher(s) for all;
    with two teachers, they take the course's classes in turn (in every section the same way), and the second
    one the practical."""
    base = generate_department(seed, n_faculty=n_faculty, n_groups=n_groups, courses_per_group=1,
                               n_lecture_rooms=n_lecture_rooms, n_labs=n_labs, calendar=calendar)
    rng = random.Random(f"shared-{seed}")
    faculty, groups = base.faculty, base.groups
    load: Counter[str] = Counter()
    sessions: list[Session] = []
    titles: dict[str, str] = {}
    for k in range(n_courses):
        course = f"C-{k + 1:02d}"
        titles[course] = COURSE_TITLES[k % len(COURSE_TITLES)]
        pair = rng.random() < co_teaching
        chosen = sorted(faculty, key=lambda f: (load[f.id], rng.random()))[: 2 if pair else 1]
        teachers = [f.id for f in chosen]
        practical = rng.random() < lab_share
        needs = rng.choice(PRACTICAL_NEEDS)
        for g in groups:
            for j in range(lectures_per_course):
                who = teachers[j % len(teachers)]
                sessions.append(Session(id=f"{course}-{g.id}-L{j + 1}", course=course, kind=SessionKind.LECTURE,
                                        faculty=who, groups=[g.id]))
                load[who] += 1
            if practical:
                who = teachers[-1]
                sessions.append(Session(id=f"{course}-{g.id}-P", course=course, kind=SessionKind.PRACTICAL,
                                        faculty=who, groups=[g.id], duration=2, room_type=RoomType.LAB,
                                        equipment=needs))
                load[who] += 2
    return base.model_copy(update={"name": f"synthetic-cse-s{seed}-shared", "sessions": sessions,
                                   "course_titles": titles})
