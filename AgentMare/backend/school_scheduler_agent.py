import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import numpy as np
import json

from custom_logging import setup_logger

# ====================== CONFIGURATION ======================
GEMINI_API_KEY = "AIzaSyDJvjsBxTcrGHRA5pRZIBL-yI1i5l4_ttU"  # Paste your API key directly here
logger = setup_logger(__name__)


# ====================== CORE SCHEDULER ======================
class SchoolScheduler:
    def __init__(self, max_hours_per_day: Dict[str, int]):
        self.classes = {}
        self.teachers = {}
        self.teacher_preferences = {}
        self.max_hours_per_day = max_hours_per_day
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        self.hours = ['8:00-8:50', '9:00-9:50', '10:00-10:50', '11:00-11:50',
                     '12:00-12:50', '13:00-13:50', '14:00-14:50']
        self.schedule = {}
        self.remaining_hours = {}
        
        self.PRIORITY_FIXED = 3
        self.PRIORITY_HIGH = 2
        self.PRIORITY_MEDIUM = 1
        self.PRIORITY_LOW = 0

    def add_class(self, class_name: str, subjects: Dict[str, int]):
        self.classes[class_name] = subjects
        self.remaining_hours[class_name] = subjects.copy()
        self.schedule[class_name] = {day: {} for day in self.days}

    def add_teacher(self, teacher_name: str, subject_classes: Dict[str, List[str]]):
        self.teachers[teacher_name] = subject_classes

    def set_teacher_preferences(self, teacher_name: str, preferred_days: List[str] = None,
                              preferred_hours: List[str] = None, day_priorities: Dict[str, int] = None,
                              hour_priorities: Dict[str, int] = None):
        if teacher_name not in self.teacher_preferences:
            self.teacher_preferences[teacher_name] = {'days': {}, 'hours': {}}
            
        for day in self.days:
            self.teacher_preferences[teacher_name]['days'][day] = self.PRIORITY_LOW
        for hour in self.hours:
            self.teacher_preferences[teacher_name]['hours'][hour] = self.PRIORITY_LOW
            
        if preferred_days:
            for day in preferred_days:
                self.teacher_preferences[teacher_name]['days'][day] = self.PRIORITY_MEDIUM
        if preferred_hours:
            for hour in preferred_hours:
                self.teacher_preferences[teacher_name]['hours'][hour] = self.PRIORITY_MEDIUM
                
        if day_priorities:
            for day, priority in day_priorities.items():
                if day in self.days:
                    self.teacher_preferences[teacher_name]['days'][day] = priority
        if hour_priorities:
            for hour, priority in hour_priorities.items():
                if hour in self.hours:
                    self.teacher_preferences[teacher_name]['hours'][hour] = priority

    def assign_slot(self, class_name: str, day: str, hour: str, subject: str, teacher: str) -> bool:
        if (day not in self.days or hour not in self.hours or 
            class_name not in self.classes or subject not in self.classes[class_name]):
            return False
            
        if (teacher not in self.teachers or 
            subject not in self.teachers[teacher] or 
            class_name not in self.teachers[teacher][subject]):
            return False
            
        for c in self.schedule:
            if day in self.schedule[c] and hour in self.schedule[c][day]:
                if self.schedule[c][day][hour][1] == teacher:
                    return False
                    
        if self.remaining_hours[class_name].get(subject, 0) <= 0:
            return False
            
        if len(self.schedule[class_name].get(day, {})) >= self.max_hours_per_day[class_name]:
            return False
            
        self.schedule[class_name][day][hour] = (subject, teacher)
        self.remaining_hours[class_name][subject] -= 1
        return True

    def auto_fill_schedule(self) -> Dict:
        stats = {'filled_slots': 0, 'remaining_subjects': {}}
        
        for class_name in self.classes:
            for day in self.days:
                for hour in self.hours:
                    if self._can_assign(class_name, day, hour):
                        teachers = self._get_available_teachers(class_name, day, hour)
                        if teachers:
                            subject = self._get_most_needed_subject(class_name)
                            if subject:
                                teacher = max(teachers, key=lambda t: self._get_teacher_score(t, day, hour))
                                if self.assign_slot(class_name, day, hour, subject, teacher):
                                    stats['filled_slots'] += 1
        
        for class_name, subjects in self.remaining_hours.items():
            stats['remaining_subjects'][class_name] = {
                subj: hours for subj, hours in subjects.items() if hours > 0
            }
            
        return stats

    def _can_assign(self, class_name: str, day: str, hour: str) -> bool:
        return (hour not in self.schedule[class_name].get(day, {}) and 
                len(self.schedule[class_name].get(day, {})) < self.max_hours_per_day[class_name])

    def _get_available_teachers(self, class_name: str, day: str, hour: str) -> List[str]:
        available = []
        for subject in self.classes[class_name]:
            if self.remaining_hours[class_name].get(subject, 0) > 0:
                for teacher, subjects in self.teachers.items():
                    if (subject in subjects and class_name in subjects[subject] and
                        not self._is_teacher_busy(teacher, day, hour)):
                        available.append(teacher)
        return available

    def _get_most_needed_subject(self, class_name: str) -> Optional[str]:
        subjects = [s for s, h in self.remaining_hours[class_name].items() if h > 0]
        if not subjects:
            return None
        return max(subjects, key=lambda s: self.remaining_hours[class_name][s])

    def _get_teacher_score(self, teacher: str, day: str, hour: str) -> int:
        if teacher not in self.teacher_preferences:
            return 0
        return (self.teacher_preferences[teacher]['days'].get(day, 0) + 
               self.teacher_preferences[teacher]['hours'].get(hour, 0))

    def _is_teacher_busy(self, teacher: str, day: str, hour: str) -> bool:
        for c in self.schedule:
            if day in self.schedule[c] and hour in self.schedule[c][day]:
                if self.schedule[c][day][hour][1] == teacher:
                    return True
        return False

    def handle_teacher_resignation(self, teacher_name: str) -> Dict:
        affected_slots = []
        for class_name, days in self.schedule.items():
            for day, hours in days.items():
                for hour, (subject, teacher) in hours.items():
                    if teacher == teacher_name:
                        affected_slots.append({
                            'class': class_name,
                            'day': day,
                            'hour': hour,
                            'subject': subject
                        })
        return {
            'resigned_teacher': teacher_name,
            'affected_slots': affected_slots,
            'affected_slots_count': len(affected_slots)
        }

    def apply_resignation_changes(self, changes: List[Dict]) -> bool:
        success = True
        for change in changes:
            if not self.assign_slot(
                change['class'], change['day'], change['hour'],
                change['subject'], change['new_teacher']
            ):
                success = False
        return success

    def get_schedule_as_text(self, class_name: str = None) -> str:
        output = []
        classes = [class_name] if class_name else self.classes.keys()
        
        for cn in classes:
            output.append(f"\nSchedule for {cn}:")
            output.append("Day/Hour".ljust(15) + "".join([h.ljust(20) for h in self.hours]))
            
            for day in self.days:
                line = [day.ljust(15)]
                for hour in self.hours:
                    if hour in self.schedule[cn].get(day, {}):
                        subj, teach = self.schedule[cn][day][hour]
                        line.append(f"{subj[:8]}({teach[:5]})".ljust(20))
                    else:
                        line.append("".ljust(20))
                output.append("".join(line))
                
        return "\n".join(output)

# ====================== GENETIC ALGORITHM ======================
class ScheduleOptimizer:
    def __init__(self, scheduler: SchoolScheduler):
        self.scheduler = scheduler
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.elitism_count = 2  # Number of best solutions to preserve

    def optimize_resignation(self, affected_slots: List[Dict]) -> List[Dict]:
        """Optimize teacher assignments for affected slots using genetic algorithm"""
        if not affected_slots:
            return []
            
        try:
            population = self._initialize_population(affected_slots)
            
            for generation in range(self.generations):
                # Evaluate fitness
                fitness = [self._evaluate_solution(ind) for ind in population]
                
                # Elitism: preserve top performers
                elite_indices = np.argpartition(fitness, -self.elitism_count)[-self.elitism_count:]
                elites = [population[i] for i in elite_indices]
                
                # Tournament selection for parents
                parents = []
                for _ in range(self.population_size - self.elitism_count):
                    candidates = random.sample(range(self.population_size), 3)
                    parents.append(population[max(candidates, key=lambda x: fitness[x])])
                
                # Crossover and mutation
                new_population = elites.copy()
                for i in range(0, len(parents), 2):
                    if i+1 < len(parents):
                        child1, child2 = self._crossover(parents[i], parents[i+1])
                        new_population.extend([child1, child2])
                
                population = [self._mutate(ind) for ind in new_population]
            
            # Return best solution
            best_idx = np.argmax([self._evaluate_solution(ind) for ind in population])
            return self._format_solution(population[best_idx])
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            return []

    def _initialize_population(self, affected_slots: List[Dict]) -> List[List[Dict]]:
        """Create initial population with diverse solutions"""
        return [self._create_random_solution(affected_slots) 
               for _ in range(self.population_size)]

    def _create_random_solution(self, slots: List[Dict]) -> List[Dict]:
        """Create one valid solution candidate"""
        solution = []
        for slot in slots:
            teachers = self._get_available_teachers_for_slot(slot)
            if teachers:
                # Weighted random choice based on teacher scores
                scores = [self._calculate_teacher_score(t, slot) for t in teachers]
                total = sum(scores)
                if total > 0:
                    probs = [s/total for s in scores]
                    chosen = np.random.choice(teachers, p=probs)
                else:
                    chosen = random.choice(teachers)
                    
                solution.append({
                    **slot,
                    'new_teacher': chosen,
                    'score': self._calculate_teacher_score(chosen, slot)
                })
        return solution

    def _get_available_teachers_for_slot(self, slot: Dict) -> List[str]:
        """Get teachers who can teach this slot"""
        available = []
        for teacher, subjects in self.scheduler.teachers.items():
            if (slot['subject'] in subjects and 
                slot['class'] in subjects[slot['subject']] and
                not self.scheduler._is_teacher_busy(teacher, slot['day'], slot['hour'])):
                available.append(teacher)
        return available

    def _calculate_teacher_score(self, teacher: str, slot: Dict) -> float:
        """Calculate fitness score for a teacher-slot assignment"""
        score = 0.0
        if teacher in self.scheduler.teacher_preferences:
            # Day preference (0-3)
            day_pref = self.scheduler.teacher_preferences[teacher]['days'].get(slot['day'], 0)
            score += day_pref * 2.0
            
            # Hour preference (0-3)
            hour_pref = self.scheduler.teacher_preferences[teacher]['hours'].get(slot['hour'], 0)
            score += hour_pref * 3.0
            
            # Penalize overworked teachers
            teacher_load = sum(
                1 for c in self.scheduler.schedule.values()
                for d in c.values()
                for h in d.values()
                if h[1] == teacher
            )
            if teacher_load > 4:  # Penalize if teaching more than 4 classes
                score -= (teacher_load - 4) * 1.5
                
        return max(score, 0.1)  # Ensure minimum score to avoid division by zero

    def _evaluate_solution(self, solution: List[Dict]) -> float:
        """Evaluate overall fitness of a solution"""
        if not solution:
            return 0.0
            
        try:
            total = sum(assign['score'] for assign in solution)
            return total / len(solution)
        except ZeroDivisionError:
            return 0.0

    def _crossover(self, parent1: List[Dict], parent2: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Uniform crossover between two parent solutions"""
        child1, child2 = [], []
        for g1, g2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(g1)
                child2.append(g2)
            else:
                child1.append(g2)
                child2.append(g1)
        return child1, child2

    def _mutate(self, individual: List[Dict]) -> List[Dict]:
        """Apply mutations to a solution with some probability"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                teachers = self._get_available_teachers_for_slot(individual[i])
                if teachers and len(teachers) > 1:  # Only mutate if alternatives exist
                    current = individual[i]['new_teacher']
                    alternatives = [t for t in teachers if t != current]
                    if alternatives:
                        individual[i]['new_teacher'] = random.choice(alternatives)
                        individual[i]['score'] = self._calculate_teacher_score(
                            individual[i]['new_teacher'], individual[i])
        return individual

    def _format_solution(self, solution: List[Dict]) -> List[Dict]:
        """Format the final solution for output"""
        return [{
            'class': s['class'],
            'day': s['day'],
            'hour': s['hour'],
            'subject': s['subject'],
            'new_teacher': s['new_teacher'],
            'score': round(s['score'], 2)
        } for s in solution]

# ====================== LLM AGENT WITH GEMINI ======================
class SchedulerAgent:
    def __init__(self, scheduler: SchoolScheduler):
        self.scheduler = scheduler
        self.optimizer = ScheduleOptimizer(scheduler)
        
        # Initialize Gemini with direct API key
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=GEMINI_API_KEY  # Using the string directly
        )
        
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True
        )

    def _create_tools(self) -> list:
        @tool
        def generate_schedule() -> str:
            """Generate a complete schedule automatically"""
            stats = self.scheduler.auto_fill_schedule()
            return (
                f"Generated schedule with {stats['filled_slots']} assignments.\n"
                f"Remaining subjects: {json.dumps(stats['remaining_subjects'], indent=2)}"
            )
        
        @tool
        def detect_conflicts() -> str:
            """Detect and report scheduling conflicts between teachers and classes"""
            conflicts = []
            # Check for teacher double bookings
            teacher_assignments = defaultdict(list)
            for class_name, days in self.scheduler.schedule.items():
                for day, hours in days.items():
                    for hour, (subject, teacher) in hours.items():
                        teacher_assignments[teacher].append((day, hour))
            
            for teacher, assignments in teacher_assignments.items():
                if len(assignments) != len(set(assignments)):
                    conflicts.append(f"Teacher {teacher} has overlapping assignments")
            
            # Check for class hour violations
            for class_name, days in self.scheduler.schedule.items():
                for day, hours in days.items():
                    if len(hours) > self.scheduler.max_hours_per_day[class_name]:
                        conflicts.append(f"Class {class_name} exceeds max hours on {day}")
            
            return "\n".join(conflicts) if conflicts else "No conflicts detected"

        @tool
        def optimize_schedule(params: str) -> str:
            """Optimize the schedule using genetic algorithms.
            Accepts JSON parameters:
            {
                "population_size": int,
                "generations": int,
                "focus_areas": list[str],
                "constraints": dict
            }
            Example:
            '{
                "population_size": 50,
                "generations": 100,
                "focus_areas": ["teacher_preferences", "balance"],
                "constraints": {"max_classes_per_day": 3}
            }'
            """
            try:
                params_dict = json.loads(params)
                self.optimizer.population_size = params_dict.get("population_size", 50)
                self.optimizer.generations = params_dict.get("generations", 100)
                
                # Create artificial "affected slots" to force full optimization
                all_slots = []
                for class_name, days in self.scheduler.schedule.items():
                    for day, hours in days.items():
                        for hour, (subject, teacher) in hours.items():
                            all_slots.append({
                                'class': class_name,
                                'day': day,
                                'hour': hour,
                                'subject': subject
                            })
                
                optimized = self.optimizer.optimize_resignation(all_slots)
                self.scheduler.apply_resignation_changes(optimized)
                
                return (f"Optimization completed with {len(optimized)} changes.\n"
                       f"Average preference score: {sum(s['score'] for s in optimized)/len(optimized):.1f}")
            except Exception as e:
                return f"Optimization failed: {str(e)}"

        @tool
        def handle_resignation(teacher_name: str) -> str:
            """Handle a teacher resignation with optimal reassignments"""
            resignation = self.scheduler.handle_teacher_resignation(teacher_name)
            if not resignation['affected_slots']:
                return f"No slots affected by {teacher_name}'s resignation"
            
            optimal = self.optimizer.optimize_resignation(resignation['affected_slots'])
            
            response = [
                f"Optimal changes for {teacher_name}'s resignation:",
                f"Affected slots: {len(optimal)}"
            ]
            for change in optimal:
                response.append(
                    f"- {change['class']} {change['day']} {change['hour']}: "
                    f"{change['subject']} → {change['new_teacher']} "
                    f"(score: {change['score']:.1f})"
                )
            
            self.scheduler.apply_resignation_changes(optimal)
            return "\n".join(response)

        @tool
        def view_schedule(class_name: str = None) -> str:
            """View the current schedule for a class or all classes"""
            return self.scheduler.get_schedule_as_text(class_name)

        @tool
        def add_class(class_info: str) -> str:
            """Add a new class to the schedule"""
            try:
                data = json.loads(class_info)
                self.scheduler.add_class(data['name'], data['subjects'])
                return f"Added class {data['name']} with {len(data['subjects'])} subjects"
            except Exception as e:
                return f"Error adding class: {str(e)}"

        @tool
        def add_teacher(teacher_info: str) -> str:
            """Add a new teacher to the schedule"""
            try:
                data = json.loads(teacher_info)
                self.scheduler.add_teacher(data['name'], data['subjects'])
                return f"Added teacher {data['name']} teaching {len(data['subjects'])} subjects"
            except Exception as e:
                return f"Error adding teacher: {str(e)}"

        return [
            generate_schedule,
            handle_resignation,
            view_schedule,
            add_class,
            add_teacher,
            detect_conflicts,
            optimize_schedule
        ]

    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert school scheduling assistant. Your job is to:
            1. Understand scheduling requests
            2. Use the appropriate tools to fulfill requests
            3. Present results clearly
            
            Always use the tools - never try to calculate schedules yourself.
            
            Current schedule info:
            - Days: {days}
            - Hours: {hours}
            - Classes: {classes}
            - Teachers: {teachers}"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        return create_tool_calling_agent(self.llm, self.tools, prompt)

    def get_schedule_info(self) -> Dict:
        return {
            "days": self.scheduler.days,
            "hours": self.scheduler.hours,
            "classes": list(self.scheduler.classes.keys()),
            "teachers": list(self.scheduler.teachers.keys()),
        }

    def run(self, query: str) -> str:
        result = self.agent_executor.invoke({
            "input": query,
            **self.get_schedule_info()
        })
        return result['output']

# ====================== EXAMPLE USAGE ======================
max_hours = {"Class 1A": 5, "Class 1B": 5, "Class 2A": 5}
scheduler = SchoolScheduler(max_hours)

# Add classes with their required subjects
scheduler.add_class("Class 1A", {
    "Math": 4,      # 4 hours/week
    "English": 4,
    "Science": 3,
    "History": 2,
    "Art": 2
})

scheduler.add_class("Class 1B", {
    "Math": 4,
    "English": 4,
    "Physics": 3,
    "Music": 2,
    "Drama": 2
})

# Add teachers with their specific class assignments
logger.debug("=== TEACHER ASSIGNMENTS ===")
scheduler.add_teacher("Ms. Math", {
    "Math": ["Class 1A", "Class 1B"]  # Teaches Math to both classes
})

scheduler.add_teacher("Mr. English", {
    "English": ["Class 1A", "Class 1B"],
    "Drama": ["Class 1B"]  # Also teaches Drama to Class 1B
})

scheduler.add_teacher("Dr. Science", {
    "Science": ["Class 1A"],
    "Physics": ["Class 1B"],
    "History": ["Class 1A"]  # Overloaded with multiple subjects
})

scheduler.add_teacher("Mrs. Arts", {
    "Art": ["Class 1A"],
    "Music": ["Class 1B"]
})

# Set teacher preferences (all competing for prime times)
scheduler.set_teacher_preferences(
    "Ms. Math",
    preferred_days=["Monday", "Wednesday"],
    preferred_hours=["8:00-8:50", "9:00-9:50"],
    day_priorities={"Monday": 3}
)

scheduler.set_teacher_preferences(
    "Mr. English",
    preferred_days=["Monday", "Tuesday"],
    preferred_hours=["8:00-8:50", "9:00-9:50"],
    day_priorities={"Monday": 3}
)

scheduler.set_teacher_preferences(
    "Dr. Science",
    preferred_days=["Monday", "Wednesday"],
    preferred_hours=["10:00-10:50", "11:00-11:50"],
    day_priorities={"Monday": 3}
)

# Create agent
agent = SchedulerAgent(scheduler)
def generate_schedule():
    logger.debug("\n=== GENERATING INITIAL SCHEDULE ===")
    return agent.run("Create a basic schedule for all classes")

def run(prompt : str) -> str:
    logger.debug(f"\n=== RUNNING PROMPT: {prompt} ===")
    return agent.run(prompt)

# Show teacher workloads before optimization
# print("\n=== TEACHER WORKLOADS BEFORE OPTIMIZATION ===")
# for teacher in scheduler.teachers:
#     workload = sum(len(classes) for subj, classes in scheduler.teachers[teacher].items())
#     print(f"{teacher}: Teaching {workload} subjects across classes")
    
    # # Force a resignation that will require optimization
    # print("\n=== CRISIS: TEACHER RESIGNATION ===")
    # print(agent.run("Dr. Science is resigning - handle this emergency"))
    
    # # Show the broken schedule
    # print("\n=== BROKEN SCHEDULE AFTER RESIGNATION ===")
    # print(agent.run("Show me all unfilled slots"))
    
    # # Add replacement teacher
    # print("\n=== ADDING REPLACEMENT TEACHER ===")
    # print(agent.run("""Add new teacher: {
    #     "name": "Mr. STEM",
    #     "subjects": {
    #         "Science": ["Class 1A"],
    #         "Physics": ["Class 1B"],
    #         "History": ["Class 1A"]
    #     }
    # }"""))
    
    # # Now optimize the complete schedule
    # print("\n=== OPTIMIZING FULL SCHEDULE ===")
    # print(agent.run("""Optimize the entire schedule with:
    # - Population size: 100
    # - Generations: 200
    # - Priorities:
    #   1. Fill all teaching slots
    #   2. Respect teacher preferences
    #   3. Balance workloads
    # """))
    
    # # Final verification
    # print("\n=== FINAL TEACHER WORKLOADS ===")
    # for teacher in scheduler.teachers:
    #     assigned_hours = sum(
    #         1 for class_sched in scheduler.schedule.values()
    #         for day in class_sched.values()
    #         for slot in day.values()
    #         if slot[1] == teacher
    #     )
    #     print(f"{teacher}: Teaching {assigned_hours} hours")

    # print("\n=== FINAL SCHEDULE ===")
    # print(agent.run("Show me Class 1A's schedule"))
    # print(agent.run("Show me Class 1B's schedule"))
