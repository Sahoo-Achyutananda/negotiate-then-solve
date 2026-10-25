import clsx from "clsx";
import { ArrowLeftRight, CalendarClock, Send, X, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import { dateOf } from "../lib/slots";
import { dayPassed, freeSlotsFor, startWeek, swappableWith } from "../lib/scheduling";
import type { TimetableEntry } from "../lib/types";
import { intent, sentence, type ActionKind, type Draft, type WeekChoice } from "./ClassActions";
import { RangeBuilder } from "./RangeBuilder";
import { Button, Skeleton } from "./ui";

/** "New request" from the Requests tab (wireframe of 24 Oct, "point scheduling"): a week, a day (or all), one of
 * your classes; then write what should happen, or cancel, swap or reschedule it with a few clicks.
 * - Reschedule: a free slot in that week for the class, and which weeks it applies to;
 * - Swap: a colleague's class it can actually be swapped with that week;
 * - Cancel: which weeks.
 * Typed text is sent as a request with the class named in it; either way it also appears in the chat.
 * A teacher can instead pick "A range of dates" (components/RangeBuilder): several classes cancelled or moved. */
export function RequestBuilder({ onClose, onSent }: { onClose: () => void; onSent: (id: string) => void }) {
  const { user } = useAuth();
  const teaching = ["faculty", "guest_faculty", "hod"].includes(user?.role ?? "");
  const group = user?.role === "student" && user.id.startsWith("ST-") ? user.id.slice(3) : null;
  const { data: inst } = useApi(() => api.instance(), []);
  const { data: cal } = useApi(() => api.calendar(), []);
  const [week, setWeek] = useState<number | null>(null);
  useEffect(() => {
    if (cal && week === null) setWeek(startWeek(cal));
  }, [cal, week]);
  const { data: tt } = useApi(() => (week === null ? Promise.resolve(null) : api.timetable({ week })), [week]);
  const all = useMemo(() => tt?.entries ?? [], [tt]);
  const [day, setDay] = useState<string | null>(null); // null: every day
  const [cls, setCls] = useState<TimetableEntry | null>(null);
  const [kind, setKind] = useState<ActionKind | null>(null);
  const [text, setText] = useState("");
  const [to, setTo] = useState<{ day: string; slot: number } | null>(null);
  const [theirs, setTheirs] = useState<string>("");
  const [when, setWhen] = useState<WeekChoice>({ mode: "this", weeks: [] });
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [range, setRange] = useState(false); // "A range of dates" instead of one class

  const mine = useMemo(
    () => all.filter((e) => (teaching ? e.faculty === user?.id : group ? e.groups.includes(group) : false)),
    [all, teaching, group, user],
  );
  const shown = mine.filter((e) => !day || e.day === day);
  // a new week or class starts the choice again
  useEffect(() => {
    setCls(null);
    setKind(null);
  }, [week]);
  useEffect(() => {
    setTo(null);
    setTheirs("");
    setWhen({ mode: "this", weeks: [] });
  }, [cls, kind]);
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  const time = (s: number) => inst?.slots.find((x) => x.slot === s)?.label ?? `slot ${s}`;
  const dayName = (d: string) => inst?.day_names[d] ?? d;
  const at = (d: string) => (cal && week ? dateOf(cal, week, d) : dayName(d));
  const free = inst && cls ? freeSlotsFor(inst, cal, week, all, cls) : [];
  const swaps = cls ? swappableWith(all, cls) : [];
  const own = teaching && !!cls && cls.faculty === user?.id; // only your own classes can be cancelled, moved, swapped
  const passed = !!cal && week !== null && week < cal.current_week;

  const draft: Draft | null = cls && kind
    ? { kind, cls, to, theirs: swaps.find((o) => o.session === theirs) ?? null, when, hard: true, reason: "" }
    : null;
  const ready = !!draft && (kind === "cancel" || (kind === "move" && !!to) || (kind === "swap" && !!draft.theirs))
    && (when.mode !== "pick" || when.weeks.length > 0);

  const send = async (how: "form" | "text") => {
    if (!inst || week === null || !cls) return;
    setSending(true);
    setError(null);
    try {
      const phrase = `${teaching ? "my" : "the"} ${cls.title} ${cls.kind} on ${dayName(cls.day)} at ${time(cls.slot)} in week ${week}`;
      const { id } = how === "form" && draft
        ? await api.submitForm(sentence(draft, inst, week), intent(draft, week))
        : await api.submit(`${text.trim().replace(/[.!?]$/, "")} (${phrase}).`);
      onSent(id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  };

  const tab = (on: boolean, off = false) =>
    clsx("shrink-0 rounded-md border px-2.5 py-1 text-[12.5px] transition-colors", off && "opacity-40",
         on ? "border-brand bg-brand/10 text-brand" : "border-line text-ink-2 hover:bg-panel-2");
  const modes: Record<ActionKind, WeekChoice["mode"][]> = { cancel: ["this", "pick"], move: ["this", "from", "pick"], swap: ["this", "every", "pick"] };
  const modeLabel = { this: `Week ${week} only`, from: `From week ${week} on`, every: "Every week", pick: "Pick weeks" };

  return (
    <div className="fixed inset-0 z-40 flex items-start justify-center overflow-y-auto bg-black/50 p-2 sm:items-center sm:p-6"
         onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="w-full max-w-3xl rounded-xl border border-line bg-bg shadow-2xl" role="dialog" aria-label="New request">
        <div className="flex items-center justify-between border-b border-line px-4 py-3">
          <p className="text-[15px] font-semibold">New request</p>
          <button onClick={onClose} className="grid size-8 place-items-center rounded-md text-ink-3 hover:bg-panel-2 hover:text-ink" aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <div className="space-y-4 p-4">
          {teaching && (
            <div className="inline-flex rounded-lg border border-line p-0.5">
              {([[false, "One class"], [true, "A range of dates"]] as const).map(([on, label]) => (
                <button key={label} onClick={() => setRange(on)}
                        className={clsx("rounded-md px-3 py-1 text-[12.5px]", range === on ? "bg-panel-2 font-medium text-ink" : "text-ink-3 hover:text-ink")}>
                  {label}
                </button>
              ))}
            </div>
          )}
          {range ? (inst && cal ? <RangeBuilder inst={inst} cal={cal} onSent={onSent} /> : <Skeleton className="h-40 w-full" />) : (<>
          {/* week → day → class */}
          <Row label="Week">
            {cal ? cal.weeks.map((w) => (
              <button key={w.week} onClick={() => setWeek(w.week)} className={tab(w.week === week, w.week < cal.current_week)}
                      title={w.week < cal.current_week ? "This week is over" : `Week ${w.week}, from ${w.monday}`}>
                {w.week}
              </button>
            )) : <Skeleton className="h-7 w-full" />}
          </Row>
          <Row label="Day">
            <button onClick={() => setDay(null)} className={tab(day === null)}>All</button>
            {(inst?.calendar.days ?? []).map((d) => (
              <button key={d} onClick={() => setDay(d)} className={tab(day === d, dayPassed(cal, week, d))}>
                {at(d)}
              </button>
            ))}
          </Row>
          <Row label={teaching ? "Your class" : "Class"}>
            {!tt ? <Skeleton className="h-8 w-full" /> : !shown.length ? (
              <p className="text-[12.5px] text-ink-3">No classes {day ? `on ${dayName(day)}` : "this week"}.</p>
            ) : shown.map((e) => (
              <button key={e.session} onClick={() => setCls(cls?.session === e.session ? null : e)} className={tab(cls?.session === e.session)}>
                <span className="font-medium">{e.title} {e.kind}</span>
                <span className="ml-1.5 text-ink-3">{day ? "" : `${at(e.day)} · `}{time(e.slot)}</span>
              </button>
            ))}
          </Row>

          {cls && (
            <div className="space-y-3 border-t border-line pt-4">
              <div className="flex items-center gap-2 rounded-xl border border-line bg-panel py-1.5 pl-3 pr-1.5 focus-within:border-brand/60">
                <textarea value={text} onChange={(e) => setText(e.target.value)} rows={1}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !e.shiftKey) {
                              e.preventDefault();
                              if (text.trim() && !sending) void send("text");
                            }
                          }}
                          placeholder={`Write what to do with ${cls.title} ${cls.kind} (${dayName(cls.day)} ${time(cls.slot)})…`}
                          title="Enter to send · Shift+Enter for a new line"
                          className="max-h-32 min-h-[1.75rem] flex-1 resize-none bg-transparent py-1 text-[13px] leading-relaxed outline-none placeholder:text-ink-3" />
                <Button variant="primary" size="sm" icon={Send} loading={sending && !kind} disabled={!text.trim() || sending}
                        onClick={() => void send("text")} aria-label="Send as a request" />
              </div>

              {own && (
                <>
                  <p className="text-center text-[11.5px] text-ink-3">or</p>
                  <div className="grid grid-cols-3 gap-2">
                    {([["cancel", "Cancel", XCircle], ["swap", "Swap", ArrowLeftRight], ["move", "Reschedule", CalendarClock]] as const).map(([k, label, Icon]) => (
                      <button key={k} onClick={() => setKind(kind === k ? null : k)}
                              className={clsx("flex items-center justify-center gap-1.5 rounded-md border px-3 py-2 text-[13px] font-medium",
                                              kind === k ? "border-brand bg-brand/10 text-brand" : "border-line hover:bg-panel-2")}>
                        <Icon size={14} /> {label}
                      </button>
                    ))}
                  </div>

                  {kind === "move" && (
                    <Row label="To">
                      {free.length ? free.map((s) => (
                        <button key={`${s.day}:${s.slot}`} onClick={() => setTo(s)} className={tab(to?.day === s.day && to.slot === s.slot)}>
                          {dayName(s.day).slice(0, 3)} {time(s.slot)}
                        </button>
                      )) : <p className="text-[12.5px] text-ink-3">No free time for this class in week {week}{passed ? " (it is over)" : ""}.</p>}
                    </Row>
                  )}
                  {kind === "swap" && (
                    <Row label="With">
                      {swaps.length ? (
                        <select value={theirs} onChange={(e) => setTheirs(e.target.value)}
                                className="h-8 min-w-0 flex-1 rounded-md border border-line bg-panel px-2 text-[12.5px] outline-none focus:border-brand/60">
                          <option value="">A colleague's class it can be swapped with…</option>
                          {swaps.map((o) => (
                            <option key={o.session} value={o.session}>
                              {o.faculty_name} · {o.title} {o.kind} · {dayName(o.day)} {time(o.slot)}
                            </option>
                          ))}
                        </select>
                      ) : <p className="text-[12.5px] text-ink-3">No colleague's class in week {week} can be swapped with this one.</p>}
                    </Row>
                  )}
                  {kind && (
                    <Row label="Weeks">
                      {modes[kind].map((m) => (
                        <button key={m} onClick={() => setWhen({ mode: m, weeks: m === "pick" ? when.weeks : [] })} className={tab(when.mode === m)}>
                          {modeLabel[m]}
                        </button>
                      ))}
                    </Row>
                  )}
                  {kind && when.mode === "pick" && cal && (
                    <Row label="">
                      {cal.weeks.filter((w) => w.week >= cal.current_week).map((w) => (
                        <button key={w.week} className={tab(when.weeks.includes(w.week))}
                                onClick={() => setWhen({ mode: "pick", weeks: when.weeks.includes(w.week) ? when.weeks.filter((x) => x !== w.week) : [...when.weeks, w.week] })}>
                          {w.week}
                        </button>
                      ))}
                    </Row>
                  )}
                  {kind && (
                    <div className="flex items-center justify-between gap-3 rounded-md bg-panel px-3 py-2">
                      <p className="text-[12.5px] text-ink-2">{draft && inst && week !== null ? sentence(draft, inst, week) : ""}</p>
                      <Button variant="primary" size="sm" loading={sending} disabled={!ready} onClick={() => void send("form")}>
                        Send
                      </Button>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
          {error && <p className="text-[12.5px] text-bad">{error}</p>}
          </>)}
        </div>
      </div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3">
      <p className="w-20 shrink-0 pt-1.5 text-[11.5px] font-medium uppercase tracking-wide text-ink-3">{label}</p>
      <div className="flex min-w-0 flex-1 gap-1.5 overflow-x-auto pb-1 sm:flex-wrap">{children}</div>
    </div>
  );
}
