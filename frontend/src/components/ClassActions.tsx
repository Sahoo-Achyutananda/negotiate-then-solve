import clsx from "clsx";
import { ArrowLeftRight, CalendarClock, PencilLine, Send, X } from "lucide-react";
import { useEffect, useRef, type ReactNode } from "react";
import type { InstanceView, TimetableEntry } from "../lib/types";
import { Button } from "./ui";

/** Requests made by clicking instead of typing: cancel a class, move it, or swap it with a colleague's.
 * The card sends the request as data (``intent``, read by no parser) together with the same request in
 * words (``sentence``), which is what people see in the conversation. */

export type ActionKind = "cancel" | "move" | "swap";
export interface Target {
  day?: string;
  slot?: number;
  part?: "morning" | "afternoon";
}
export interface WeekChoice {
  mode: "this" | "from" | "every" | "pick";
  weeks: number[];
}
export interface Draft {
  kind: ActionKind;
  cls: TimetableEntry;
  to: Target | null;
  theirs: TimetableEntry | null;
  when: WeekChoice;
  hard: boolean;
  reason: string;
}

const WHEN: Record<ActionKind, WeekChoice["mode"][]> = { cancel: ["this", "pick"], move: ["this", "from", "pick"], swap: ["this", "every", "pick"] };

const listed = (xs: (string | number)[]) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

/** The request in words: what the card sends. */
export function sentence(d: Draft, inst: InstanceView, week: number): string {
  const day = (x: string) => inst.day_names[x] ?? x;
  const time = (s: number) => inst.slots.find((x) => x.slot === s)?.label ?? `slot ${s}`;
  const cls = `my ${d.cls.title} ${d.cls.kind} on ${day(d.cls.day)} at ${time(d.cls.slot)}`;
  const weeks = [...d.when.weeks].sort((a, b) => a - b);
  const when =
    d.when.mode === "this"
      ? `in week ${week} only`
      : d.when.mode === "from"
        ? `from week ${week} onwards`
        : d.when.mode === "every"
          ? "every week"
          : weeks.length === 1
            ? `in week ${weeks[0]}`
            : `in weeks ${listed(weeks)}`;
  const reason = d.reason.trim() ? ` The reason: ${d.reason.trim().replace(/[.]$/, "")}.` : "";
  if (d.kind === "cancel") return `Please cancel ${cls} ${when}.${reason}`;
  if (d.kind === "swap" && d.theirs) {
    const o = d.theirs;
    return `Please swap ${cls} with ${o.faculty_name}'s ${o.title} ${o.kind} on ${day(o.day)} at ${time(o.slot)}, ${when}.${reason}`;
  }
  const t = d.to ?? {};
  const to =
    t.slot != null && t.day
      ? `to ${day(t.day)} at ${time(t.slot)}`
      : t.part
        ? `to any time ${t.part === "morning" ? "before" : "after"} lunch${t.day ? ` on ${day(t.day)}` : ""}`
        : `to any time on ${day(t.day!)}`;
  return `Please move ${cls} ${to}, ${when}${d.hard ? "; it has to be then" : ", if possible"}.${reason}`;
}

const FIRST_HOUR = 9; // slot 0 starts at 9:00 (language.corpus.FIRST_SLOT_HOUR)
export const clock = (slot: number) => `${String(FIRST_HOUR + slot).padStart(2, "0")}:00`;

/** The request as the pipeline takes it: one intent of ``language.intents`` (cancel, move or swap). */
export function intent(d: Draft, week: number): Record<string, unknown> {
  const weeks = [...d.when.weeks].sort((a, b) => a - b);
  const period = d.when.mode === "this" ? { weeks: [week] } : d.when.mode === "from" ? { from_week: week } : d.when.mode === "every" ? {} : { weeks };
  const justification = d.reason.trim() ? "stated" : "none";
  const which = { session: d.cls.session };
  if (d.kind === "cancel") return { kind: "cancel", which, period, justification };
  if (d.kind === "swap") return { kind: "swap", mine: which, theirs: { session: d.theirs!.session, owner: d.theirs!.faculty }, period, justification };
  const t = d.to ?? {};
  const to = t.slot != null && t.day ? { days: [t.day], at: [clock(t.slot)] } : { ...(t.part ? { part: t.part } : {}), ...(t.day ? { days: [t.day] } : {}) };
  return { kind: "move", which, to, period, hard: d.hard, justification };
}

export function ready(d: Draft): boolean {
  if (d.when.mode === "pick" && !d.when.weeks.length) return false;
  if (d.kind === "move") return !!d.to && (d.to.slot != null || !!d.to.part || !!d.to.day);
  if (d.kind === "swap") return !!d.theirs;
  return true;
}

/** What can be done with a class, at the place it was clicked. */
export function ClassMenu({ at, cls, own, onPick, onClose }: {
  at: { x: number; y: number };
  cls: TimetableEntry;
  /** The person teaches it: they may cancel, move or swap it. */
  own: boolean;
  onPick: (kind: ActionKind | "write") => void;
  onClose: () => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const click = (e: MouseEvent) => !box.current?.contains(e.target as Node) && onClose();
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("mousedown", click);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("mousedown", click);
      window.removeEventListener("keydown", key);
    };
  }, [onClose]);
  const left = Math.min(at.x, window.innerWidth - 248);
  const top = Math.min(at.y, window.innerHeight - (own ? 200 : 100));
  const item = (kind: ActionKind | "write", icon: ReactNode, label: string) => (
    <button onClick={() => onPick(kind)} className="flex w-full items-center gap-2.5 rounded px-2.5 py-1.5 text-left text-[13px] hover:bg-panel-2">
      {icon}
      {label}
    </button>
  );
  return (
    <div ref={box} style={{ left, top }} className="fixed z-40 w-60 rounded-lg border border-line bg-panel p-1 shadow-lg" role="menu">
      <div className="px-2.5 pb-1.5 pt-1">
        <p className="truncate text-[13px] font-semibold">
          {cls.title} <span className="font-normal text-ink-3">{cls.kind}</span>
        </p>
        <p className="truncate text-[11.5px] text-ink-3">
          {cls.room_name} · {cls.group_names.join(", ")}
        </p>
      </div>
      {own && (
        <div className="border-t border-line pt-1">
          {item("cancel", <X size={14} className="text-bad" />, "Request cancellation")}
          {item("move", <CalendarClock size={14} className="text-brand" />, "Request rescheduling")}
          {item("swap", <ArrowLeftRight size={14} className="text-info" />, "Swap with a colleague")}
        </div>
      )}
      <div className="mt-1 border-t border-line pt-1">{item("write", <PencilLine size={14} className="text-ink-3" />, "Write about this class")}</div>
    </div>
  );
}

/** The card in the panel while a click-made request is being put together. */
export function ActionCard({ d, set, inst, week, weeks, current, sending, error, onCancel, onSend }: {
  d: Draft;
  set: (d: Draft) => void;
  inst: InstanceView;
  week: number;
  weeks: number;
  /** The teaching week now: earlier weeks have passed and cannot be asked for. */
  current: number;
  sending: boolean;
  error: string | null;
  onCancel: () => void;
  onSend: () => void;
}) {
  const day = (x: string) => inst.day_names[x] ?? x;
  const time = (s: number) => inst.slots.find((x) => x.slot === s)?.label ?? `slot ${s}`;
  const title = { cancel: "Cancel a class", move: "Reschedule a class", swap: "Swap with a colleague" }[d.kind];
  const whenLabel = { this: `Week ${week} only`, from: `From week ${week} on`, every: "Every week", pick: "Pick weeks" };
  const seg = (on: boolean) => clsx("rounded-md border px-2.5 py-1 text-[12.5px]", on ? "border-brand bg-brand/10 text-brand" : "border-line text-ink-2 hover:bg-panel-2");
  const t = d.to;

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        <div>
          <p className="text-[14px] font-semibold">{title}</p>
          <p className="mt-0.5 text-[12.5px] text-ink-3">
            {d.cls.title} {d.cls.kind} · {day(d.cls.day)} {time(d.cls.slot)} · {d.cls.group_names.join(", ")}
          </p>
        </div>

        {d.kind === "move" && (
          <Field label="To">
            {t?.slot != null && t.day ? (
              <p className="text-[13px]">
                {day(t.day)} {time(t.slot)}{" "}
                <button onClick={() => set({ ...d, to: null })} className="ml-1 text-[12px] text-brand hover:underline">
                  change
                </button>
              </p>
            ) : (
              <div className="space-y-2">
                <p className="text-[12.5px] text-ink-2">Click a green slot on the timetable, or:</p>
                <div className="flex flex-wrap gap-1.5">
                  {(["morning", "afternoon"] as const).map((p) => (
                    <button key={p} onClick={() => set({ ...d, to: { ...t, part: t?.part === p ? undefined : p, slot: undefined } })} className={seg(t?.part === p)}>
                      Any {p}
                    </button>
                  ))}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {inst.calendar.days.map((x) => (
                    <button key={x} onClick={() => set({ ...d, to: { ...t, day: t?.day === x ? undefined : x, slot: undefined } })} className={seg(t?.day === x)}>
                      {x}
                    </button>
                  ))}
                </div>
                <p className="text-[11.5px] text-ink-3">A part of the day, a day, or both.</p>
              </div>
            )}
          </Field>
        )}

        {d.kind === "swap" && (
          <Field label="You take">
            {d.theirs ? (
              <p className="text-[13px]">
                {d.theirs.faculty_name}'s {d.theirs.title} {d.theirs.kind} · {day(d.theirs.day)} {time(d.theirs.slot)}{" "}
                <button onClick={() => set({ ...d, theirs: null })} className="ml-1 text-[12px] text-brand hover:underline">
                  change
                </button>
              </p>
            ) : (
              <p className="text-[12.5px] text-ink-2">Click a colleague's class on the timetable. Classes that cannot be swapped are faded.</p>
            )}
          </Field>
        )}

        <Field label="When">
          <div className="flex flex-wrap gap-1.5">
            {WHEN[d.kind].map((m) => {
              const passed = (m === "this" || m === "from") && week < current;
              return (
                <button
                  key={m}
                  disabled={passed}
                  title={passed ? `Week ${week} has passed` : undefined}
                  onClick={() => set({ ...d, when: { mode: m, weeks: m === "pick" ? (d.when.weeks.length ? d.when.weeks : week >= current ? [week] : []) : [] } })}
                  className={clsx(seg(d.when.mode === m), passed && "cursor-not-allowed opacity-40")}
                >
                  {whenLabel[m]}
                </button>
              );
            })}
          </div>
          {d.when.mode === "pick" && (
            <div className="mt-2 flex flex-wrap gap-1">
              {Array.from({ length: weeks }, (_, i) => i + 1).map((w) => {
                const on = d.when.weeks.includes(w);
                const passed = w < current;
                return (
                  <button
                    key={w}
                    disabled={passed}
                    title={passed ? `Week ${w} has passed` : undefined}
                    onClick={() => set({ ...d, when: { ...d.when, weeks: on ? d.when.weeks.filter((x) => x !== w) : [...d.when.weeks, w] } })}
                    className={clsx(
                      "grid h-7 w-7 place-items-center rounded text-[12px] tabular-nums",
                      on ? "bg-brand font-semibold text-panel" : passed ? "cursor-not-allowed text-ink-3/40" : "border border-line hover:bg-panel-2",
                    )}
                  >
                    {w}
                  </button>
                );
              })}
            </div>
          )}
        </Field>

        {d.kind === "move" && (
          <Field label="How">
            <div className="flex flex-wrap gap-1.5">
              <button onClick={() => set({ ...d, hard: true })} className={seg(d.hard)}>
                Must be then
              </button>
              <button onClick={() => set({ ...d, hard: false })} className={seg(!d.hard)}>
                If possible
              </button>
            </div>
            {!d.hard && <p className="mt-1.5 text-[11.5px] text-ink-3">Only if no other class has to move for it, so it may not happen.</p>}
          </Field>
        )}

        <Field label="Reason (optional)">
          <input
            value={d.reason}
            onChange={(e) => set({ ...d, reason: e.target.value })}
            placeholder={d.kind === "cancel" ? "e.g. conference, leave" : "e.g. exam duty"}
            className="h-8 w-full rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none focus:border-brand/60"
          />
          <p className="mt-1 text-[11.5px] text-ink-3">Kept private: only the fact that a reason was given is used.</p>
        </Field>

        {d.kind === "cancel" && <Note>A cancelled class needs a make-up class within two weeks.</Note>}
        {d.kind === "swap" && <Note>Your colleague will be asked to agree, and the timetable office is told before the swap takes effect.</Note>}
        {d.kind === "move" && <Note>Green slots are free for you and your section in the published timetable. The final check is the solver's.</Note>}
      </div>

      <div className="border-t border-line p-3">
        {error && <p className="mb-2 text-[12.5px] text-bad">{error}</p>}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" icon={Send} loading={sending} disabled={!ready(d)} onClick={onSend}>
            Send request
          </Button>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-ink-3">{label}</p>
      {children}
    </div>
  );
}

function Note({ children }: { children: ReactNode }) {
  return <p className="rounded-md bg-panel-2/70 px-3 py-2 text-[12px] text-ink-2">{children}</p>;
}
