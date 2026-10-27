// Shapes returned by the Python API (src/nts/web.py).

export type Role = "coordinator" | "dean" | "hod" | "faculty" | "guest_faculty" | "lab_incharge" | "exam_cell" | "student";

export interface Persona {
  id: string;
  name: string;
  role: Role;
  email?: string;
  /** Views this person may open, decided by the server (``VIEWS`` in nts/web.py). */
  views?: View[];
}

export type View = "home" | "requests" | "timetable" | "handbook" | "how" | "new" | "inbox" | "approvals" | "history" | "documents" | "semester" | "prefs" | "clubs" | "fairness" | "graph" | "health" | "experiments" | "study" | "settings";

export type Status =
  | "received"
  | "classified"
  | "refused"
  | "policy_checked"
  | "denied"
  | "clarification_requested"
  | "compiled"
  | "solved"
  | "negotiating"
  | "fairness_audited"
  | "awaiting_approval"
  | "published"
  | "escalated"
  | "answered"
  | "forwarded"
  | "withdrawn";

export interface EventRow {
  n: number;
  case: string | null;
  at: string;
  kind: string;
  [key: string]: unknown;
}

export interface CaseSummary {
  escalated_to?: string | null;
  /** Who a forwarded message went to: "coordinator", or "office" (outside the timetable). */
  forwarded_to?: string | null;
  /** The coordinator has answered it. */
  handled?: boolean;
  id: string;
  sender: string;
  sender_name: string;
  role: Role;
  channel: string;
  text: string;
  received_at: string;
  /** Classes missed in an absence: true, make-up classes; false, cancelled; null, not asked (moved). */
  makeup?: boolean | null;
  status: Status;
  route: string;
  type: string | null;
  rounds: number;
  running: boolean;
  /** In the person's own list (scope "mine"): they may withdraw it now. */
  can_withdraw?: boolean;
  /** In the person's list: it waits on their answer (a question back, or options a colleague's request sent them). */
  waits_on_you?: boolean;
}

export interface ConstraintView {
  id: string;
  type: string;
  hard: boolean;
  tier: number;
  tier_name: string;
  /** plain-language tier and who may change it (web/wording.py) */
  tier_label?: string;
  tier_who?: string;
  owner: string | null;
  owner_name: string;
  text: string;
  when: { days: string[] | null; slots: number[] | null; weeks: number[] | null };
  justification: string;
  source: { request: string | null; rule: string | null };
}

export interface PlacementView {
  day: string;
  slot: number;
  room: string;
  text: string;
  time: string;
}

export interface OfferView {
  key: string;
  drop: string[];
  cost: number;
  moved: number;
  verified: boolean;
  /** ``date`` (ISO) is the first day it happens, in week ``week``; ``weeks`` is how many weeks it covers. */
  placements: (PlacementView & { session: string; session_name: string; date?: string; week?: number; weeks?: number })[];
}

export interface Claim {
  text: string;
  facts: string[];
  supported: boolean;
}

export interface MessageView {
  round: number;
  to: string;
  to_name: string;
  mus: string[];
  offers: OfferView[];
  text: string;
  explanation_mode: string;
  faithfulness: number;
  leaks: string[];
  rejected_claims: number;
  facts: { id: string; text: string }[];
  claims: Claim[];
}

export interface ReplyView {
  decision: "accept" | "reject" | "counter" | "no_reply" | null;
  choice?: string;
  counter_days?: string[];
  counter_slots?: number[];
  text?: string;
}

export interface InboxItem {
  id: string;
  case: string | null;
  to: string;
  to_name: string;
  created: string;
  deadline: string;
  message: MessageView;
  reply: ReplyView | null;
  answered_by: string | null;
}

/** One entry in a person's inbox: a negotiation message, a change to their timetable, news of a request
 * they sent, or something waiting on them. */
export interface FeedItem {
  id: string;
  /** The last event of it; read once the person has opened it at this event. */
  n: number;
  kind: "negotiation" | "change" | "request" | "action" | "alert";
  case: string | null;
  title: string;
  text: string;
  at: string;
  from: string | null;
  from_name: string | null;
  status: string | null;
  request: string | null;
  needs_you: boolean;
  unread: boolean;
  version?: number | null;
  action?: "approve" | "decide" | "reply";
  message?: InboxItem;
}

export interface DiffRow {
  session: string;
  session_name: string;
  /** An extra class, held in this week only. */
  extra?: boolean;
  faculty: string | null;
  faculty_name: string | null;
  groups: string[];
  before: PlacementView | null;
  after: PlacementView | null;
}

export interface CaseDetail extends CaseSummary {
  routing: {
    request_type: string;
    type_p: number;
    action: string;
    action_p: number;
    tau: number;
    fast_path: boolean;
    latency_ms: number;
  } | Record<string, never>;
  timings: Record<string, number>;
  parse: null | {
    action: string;
    refusal: string | null;
    errors: string[];
    output: Record<string, unknown> | null;
    constraints: ConstraintView[];
  };
  policy: null | {
    verdict: "allowed" | "needs_approval" | "forbidden";
    cited: { id: string; cite: string }[];
    obligations: string[];
    retrieved: string[];
    hallucinated: string[];
    explanation: string;
    alternative: string | null;
    answer: string | null;
    uncited_escalation: boolean;
    mode: string;
  };
  outcome: null | {
    status: "feasible" | "agreed" | "escalated";
    step: number;
    rounds: number;
    relaxed: string[];
    mus_log: ConstraintView[][];
    messages: MessageView[];
    replies: ReplyView[];
    concessions: { stakeholder: string; name: string; constraint_id: string; credit: number }[];
    notices: Record<string, string[]>;
    escalation: null | {
      to: string;
      to_name: string;
      reason: string;
      text: string;
      /** What granting would set aside (physical facts never are). */
      set_aside: ConstraintView[];
      /** Why granting would not help; null when it can be granted. */
      cannot_grant: string | null;
    };
    solver: null | { status: string; wall_time: number; moved: number | null; soft_violations: Record<string, number> };
  };
  /** builds_on: an earlier change still waiting for approval that this one is built on (approved first). */
  proposal: null | { version: number; week: number | null; diff: DiffRow[]; builds_on?: string | null };
  fairness: { gini_before?: number; gini_after?: number; conceded?: string[]; preferences_lost?: string[] };
  /** How a person with the authority settled its escalation. */
  decision?: null | { by: string; by_name: string; granted: boolean; note: string; set_aside?: string[] };
  /** This viewer is the one the escalation was sent to, and it is still open. */
  can_decide?: boolean;
  /** The coordinator's answer to a forwarded message. */
  answer?: null | { by: string; by_name: string; note: string; at: string; acting?: string };
  /** This viewer is the coordinator and the forwarded message is still unanswered. */
  can_handle?: boolean;
  /** This viewer sent it, and nothing it asked for is in the published timetable yet. */
  can_withdraw?: boolean;
  /** An extra class: which class and week, and where it is held once a free time was found. */
  extra?: null | { template: string | null; week: number | null; session?: string; label?: string; placement: PlacementView | null };
  reply: string;
  notices: Record<string, string>;
  inbox: InboxItem[];
  events: EventRow[];
}

export interface Overview {
  department: string;
  semester: string;
  seeding: boolean;
  counts: Partial<Record<Status, number>>;
  total: number;
  pending_approvals: number;
  my_decisions?: number;
  my_forwarded?: number;
  my_inbox: number;
  /** requests waiting on this person's answer (the Requests tab's count) */
  my_waiting?: number;
  published_version: number | null;
  gini: number;
  negotiations: { agreed: number; escalated: number; feasible: number; mean_rounds: number | null };
  recent: EventRow[];
  models: { parser: string; policy: string };
  model?: { up: boolean | null; error: string | null; since: number; mode: string } | null; // null: offline mode
  sessions: number;
  faculty: number;
  rooms: number;
  groups: number;
  test_run: TestRun | null;
}

export interface TestRow {
  case: string;
  corpus_id: string;
  sender: string;
  text: string;
  expected: string;
  got: string | null;
  match: boolean;
  status: Status;
  error: string | null;
}

export interface TestRun {
  n: number;
  rows: TestRow[];
}

export interface TimetableEntry {
  session: string;
  /** An extra class, held in this week only. */
  extra?: boolean;
  course: string;
  title: string;
  kind: "lecture" | "tutorial" | "practical";
  faculty: string;
  faculty_name: string;
  groups: string[];
  group_names: string[];
  duration: number;
  day: string;
  slot: number;
  room: string;
  room_name: string;
}

export interface Timetable {
  version: number;
  week: number | null;
  approved_by: string | null;
  parent: number | null;
  entries: TimetableEntry[];
  changed: string[];
  weeks: number[]; // weeks with their own changes
  cancelled: { session: string; session_name: string; faculty_name: string; groups: string[] }[]; // not held this week
}

export interface WeeklyChange {
  case: string;
  sender_name: string;
  text: string;
  received_at: string;
  version: number | null; // null: recorded, nothing had to move
  week: number | null; // null: every week
  moved: DiffRow[];
  cancelled: DiffRow[];
  decided_by: string | null;
}

export interface CalendarView {
  days: string[];
  day_names: Record<string, string>;
  current_week: number;
  weeks: {
    week: number;
    monday: string;
    own_changes: boolean;
    class_days: string[];
    cancelled_days: string[];
    classes: { day: string; slot: number; time: string; room: string; session: string; title: string; kind: string; cancelled: boolean }[];
  }[];
}

export interface VersionRow {
  version: number;
  parent: number | null;
  approved_by: string | null;
  approved_by_name: string | null;
  published: boolean;
  case: string | null;
  created_at: string;
  week: number | null;
  carried: { version: number; week: number }[]; // the same change applied to weeks with their own repair
}

export interface InstanceView {
  name: string;
  calendar: { days: string[]; slots_per_day: number; lunch_slot: number | null; weeks: number };
  slots: { slot: number; label: string }[];
  day_names: Record<string, string>;
  faculty: { id: string; name: string; role: Role; reports_to: string | null; email: string | null }[];
  groups: { id: string; name: string; size: number }[];
  rooms: { id: string; name: string; capacity: number; type: string; equipment: string[] }[];
  sessions: { id: string; course: string; kind: string; faculty: string; groups: string[]; duration: number; title: string }[];
  policy: { max_consecutive: number | null; lunch_break: boolean };
}

export interface Ledger {
  semester: string;
  gini: number;
  decay: number;
  entries: {
    stakeholder: string;
    name: string;
    constraint_id: string;
    semester: string;
    credit: number;
    /** The semester's Gini just after this concession. */
    gini_after: number;
    /** The request whose constraint was given up. */
    case: string | null;
  }[];
  stakeholders: { id: string; name: string; credit: number; concessions: number }[];
}

export interface GraphData {
  nodes: { id: string; kind: string; label: string; role?: string; type?: string; tier?: number; balance?: number }[];
  edges: { source: string; target: string; rel: string }[];
}

export interface Rule {
  id: string;
  number: string;
  title: string;
  text: string;
  source: string;
  pages: number[];
  method: string; // "markdown", "html", "text layer" or "ocr: <backend>"
  score?: number;
}

export interface PolicyDocument {
  name: string;
  kind: "pdf" | "html" | "image" | "markdown";
  pages: number;
  methods: Record<string, number>;
  ocr_confidence: number | null;
  rules: string[];
  seconds: number;
  error: string | null;
}

export interface PolicyDocuments {
  ocr: { backend: string; model: string };
  accepts: string[];
  documents: PolicyDocument[];
}

export interface Transparency {
  tiers: { tier: number; name: string; who: string; examples: string; relaxable_in_negotiation: boolean }[];
  weights: Record<string, number>;
  role_authority: Record<string, number>;
  justification: Record<string, number>;
  ladder: { step: number; name: string; text: string }[];
  negotiation: Record<string, number>;
  system_one: { trained_on?: number; val_action_accuracy?: number; tau?: number };
  models: { parser: string; policy: string };
  safety: string[];
}

export interface Observability {
  uptime_s: number;
  models: { parser: string; policy: string };
  llm: { model: string; used_today: number; rpd: number | null; rpm: number | null; cached_responses: number }[];
  stages: Record<string, { n: number; p50: number | null; p95: number | null; max: number | null }>;
  solver: { n: number; p50: number | null; p95: number | null };
  routing: { counts: Record<string, number>; tau: number; system_one_latency_ms_p50: number | null; fast_path_share: number | null };
  explanations: { messages: number; faithfulness: number | null; leaks: number; rejected_claims: number };
  events_by_kind: Record<string, number>;
  errors: EventRow[];
  safety: { refused: number; denied: number; unapproved_publishes: number; leaks: number };
  threads_running: number;
}

export interface NegotiationSummary {
  n: number;
  correct_outcome: number | null;
  validity_of_timetables: number | null;
  timetables_produced: number;
  agreement_rate: number | null;
  rounds_mean: number | null;
  rounds_median: number | null;
  escalation_rate: number | null;
  rounds_ci: [number, number] | null;
  objective_gap_abs_mean: number | null;
  objective_gap_mean: number | null;
  objective_gap_ci: [number, number] | null;
  tier_worse_than_oracle: number | null;
  escalation_precision: number | null;
  escalation_recall: number | null;
  escalation_reasons: Record<string, number>;
  concession_gini: number | null;
  burden_max: number | null;
  burden_cv: number | null;
  burden_total: number | null;
  first_proposal_acceptance: number | null;
  explanation_faithfulness: number | null;
  private_leaks: number;
  imposed_acceptable: number | null;
  status: Record<string, number>;
}

export interface PairedTest {
  pairs: number;
  correct_outcome_mcnemar: { only_first: number; only_second: number; p: number };
  objective_wilcoxon: { n: number; p: number; median_diff: number };
}

export interface ModelRow {
  model: string;
  reply_tool: number | null;
  reply_args: number | null;
  to_coordinator: number | null;
  parse_action: number | null;
  compile_exact: number | null;
  injections: string | null;
  latency_p50: number | null;
  policy_allow_deny: number | null;
  deny_recall: number | null;
  makeup_recall: number | null;
  swap_pairs: number | null;
  swap_wrong: number | null;
}

export type Experiment =
  | { id: string; title: string; subtitle?: string; at: string; kind: "negotiation"; configs: Record<string, NegotiationSummary>; paired?: Record<string, PairedTest>; models?: { configs: string[]; model: string }[]; simulator?: string | null }
  | { id: string; title: string; subtitle?: string; at: string; kind: "models"; rows: ModelRow[] }
  | { id: string; title: string; at: string; kind: "safety"; summary: Record<string, unknown> };

// -- semester timetable ---------------------------------------------------------

export interface SemWindow {
  days: string[];
  start: string;
  end: string;
  reason: string;
}

export interface SemCalendar {
  days: string[];
  mirror: Record<string, string>;
  lecture_slots: [string, string][];
  tutorial_slots: [string, string][];
  lab_starts: string[];
  blocked: SemWindow[];
  club_hours: SemWindow[];
  weeks: number;
  evening_slot: number;
}

export interface SemRoom {
  id: string;
  capacity: number;
  type: string;
}

export interface SemCohort {
  id: string;
  name: string;
  size: number;
  courses: string[];
  electives: string[];
  /** Each elective slot and the pools its course can come from ([] for honours projects). */
  elective_pools?: { slot: string; pools: { id: string; name: string; courses: string[] }[] }[];
}

export interface SemNeed {
  kind: "lecture" | "tutorial" | "lab";
  label: string;
  rooms: number[];
  room_type: string;
  options: number;
  pair: boolean;
}

export interface SemCourse {
  code: string;
  name: string;
  L: number;
  T: number;
  P: number;
  C: number;
  faculty: string[];
  half: string;
  cap: number | null;
  cohorts: string[];
  pools: string[];
  scheduled: boolean;
  note: string;
  needs: SemNeed[];
}

export interface SemPreference {
  id: string;
  target: "faculty" | "course" | "cohort";
  who: string;
  mode: "avoid" | "prefer";
  days: string[] | null;
  start: string | null;
  end: string | null;
  hard: boolean;
  weight: number;
  text: string;
  source: string;
}

export interface SemReport {
  status: string;
  seconds: number;
  components: number;
  meetings: number;
  hard_violations: string[];
  preferences: { id: string; text: string; met: boolean; hard: boolean; sessions: number }[];
  pool_clashes: number;
  evening_sessions: number;
  moved: number;
  notes: string[];
  room_use?: Record<string, number>;
}

export interface SemVersionMeta {
  version: number;
  created: string;
  kind: "build" | "change";
  note: string;
  published: boolean;
  based_on: number | null;
  moved: number;
  report: SemReport;
}

export interface SemesterOverview {
  loaded: boolean;
  source: string | null;
  semester_start: string;
  calendar: SemCalendar;
  published: number | null;
  cohorts: SemCohort[];
  rooms: SemRoom[];
  courses?: SemCourse[];
  catalog?: { code: string; name: string; faculty: string[]; cohorts: string[]; pools: string[]; L: number; T: number; P: number; half: string }[];
  pools?: { id: string; name: string; courses: string[]; cap: number | null }[];
  warnings?: string[];
  preferences?: SemPreference[];
  closures?: { room: string; window: SemWindow; text: string }[];
  versions?: SemVersionMeta[];
  sample_available?: boolean;
  sample_is_demo?: boolean;
  demo_loaded?: boolean;
}

export interface SemMeeting {
  component: string;
  course: string;
  name: string;
  kind: "lecture" | "tutorial" | "lab";
  label: string;
  day: string;
  start: string;
  end: string;
  slot: number | null;
  rooms: string[];
  faculty: string[];
  cohorts: string[];
  half: string;
}

export interface SemDiffRow {
  component: string;
  course: string;
  name: string;
  label: string;
  before: string;
  after: string;
}

export interface SemTimetable {
  version: number;
  published: boolean;
  kind: string;
  note: string;
  diff: SemDiffRow[];
  meetings: SemMeeting[];
  report: SemReport;
}

export interface SemStatus {
  running: boolean;
  kind?: string;
  note?: string;
  elapsed?: number;
  error?: string | null;
  version?: number | null;
}

export interface ParsedInput {
  ok: boolean;
  error?: string;
  summary?: string[];
  preferences?: SemPreference[];
  closures?: ({ room: string } & SemWindow)[];
  status?: SemStatus;
}

export interface ClubRequest {
  club: string;
  activity: string;
  date: string;
  start: string;
  end: string;
  attendees: number;
  cohorts: string[];
  room: string | null;
  requested_by?: string;
}

export interface ClubDecision {
  id: string;
  request: ClubRequest;
  status: "booked" | "needs_approval" | "rejected";
  room: string | null;
  day: string;
  week: number | null;
  checks: { name: string; ok: boolean; detail: string }[];
  alternatives: { date: string; day: string; start: string; end: string; room: string }[];
  rules: { id: string; title: string; source: string; text: string }[];
  decided_at: string;
}

/** POST /api/ask: a question about the timetable answered from it, or "not_a_question" (send it as a request). */
export interface AskAnswer {
  kind: "timetable" | "not_a_question";
  answer?: string;
  /** Who read the question: the model (the lookup adapter), the regex, or the request words that ruled it out. */
  reader?: "model" | "regex" | "request words";
  call?: { tool: string; args: Record<string, unknown>; week: number; confident: boolean };
  /** For free-hour questions: the free hours, for the timetable to shade. */
  free?: { day: string; slot: number }[];
  /** The part of the timetable the answer is about, to draw beside it; null when there is nothing to draw. */
  view?: SliceView | null;
}

/** What a timetable slice shows (agents/views.py). Two layouts:
 * "week": one timetable, days across and hours down (free hours, classes, the next class, a week's changes);
 * "rows": one row per room or person, hours across, for one day (which rooms are free, who is free). */
export type SliceView = WeekSlice | RowsSlice;

export interface WeekSlice {
  layout: "week";
  week: number;
  /** The columns, in calendar order: the days asked about, or the days with something on them. */
  days: string[];
  /** The rows: the hours from the first to the last cell (slot numbers), lunch included when it falls between. */
  slots: number[];
  lunch: number | null;
  cells: SliceCell[];
}

export interface SliceCell {
  day: string;
  slot: number;
  /** Hours it covers from ``slot`` (a 2-hour practical: 2). */
  span: number;
  /** free / busy: an hour asked about; class: a class found; cancelled / moved_from / moved_to / extra: a change. */
  kind: "free" | "busy" | "class" | "cancelled" | "moved_from" | "moved_to" | "extra";
  label?: string; // classes: "Data Structures lecture"
  sub?: string; // classes: "Room 102 · Section 1 · Dr. Menon"
  session?: string;
}

export interface RowsSlice {
  layout: "rows";
  week: number;
  day: string;
  /** The columns: the hours to show (slot numbers). */
  slots: number[];
  lunch: number | null;
  /** One row per room or person, with the hours (slot numbers) they are free on ``day``. */
  rows: { name: string; free: number[] }[];
}

/** One message of the person's chat (GET/POST /api/chat, POST /api/chat/confirm; CHAT-PLAN.md). */
export type ChatMessage =
  /** what the person wrote; ``read_as``: a follow-up as it was read, made to stand on its own ("and Friday?" →
   *  "when is Dr. Rao free on Friday?"), when it was rewritten */
  | (ChatBase & { role: "you"; kind: "text"; text: string; read_as?: string })
  | (ChatBase & { role: "assistant"; kind: "lookup" } & Omit<AskAnswer, "kind">)
  /** a question about the rules, answered from them: every rule cited (none: "I couldn't find a rule"). */
  | (ChatBase & { role: "assistant"; kind: "rule"; answer: string; rules: { id: string; title: string }[] })
  /** "send this as a request?": nothing sent until answered (POST /api/chat/confirm). */
  | (ChatBase & { role: "assistant"; kind: "confirm"; text: string; makeup?: boolean | null; answered?: ConfirmAs })
  /** a request, once sent: its number (what happens next is on its page) */
  | (ChatBase & { role: "assistant"; kind: "request"; case: string })
  | (ChatBase & { role: "assistant"; kind: "other"; text: string });

export type ConfirmAs = "request" | "question" | "cancel";

interface ChatBase {
  n: number;
  at: string;
}
