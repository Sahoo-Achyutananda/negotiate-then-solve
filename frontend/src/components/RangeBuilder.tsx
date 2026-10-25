import clsx from "clsx";
import { ArrowLeft, CalendarClock, XCircle } from "lucide-react";
import { useMemo, useState } from "react";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { dateOf } from "../lib/slots";
import { startWeek } from "../lib/scheduling";
import type { CalendarView, InstanceView } from "../lib/types";
import { clock } from "./ClassActions";
import { Button } from "./ui";

/** One of the teacher's classes on a date of the range. */
interface Item {
  key: string; // week:session
  week: number;
  day: string;
  date: string; // YYYY-MM-DD
  slot: number;
  session: string;
  title: string;
  kind: string;
  room: string;
}

type Action = "cancel" | "move";

const iso = (d: Date) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
const plus = (day: string, n: number) => {
  const d = new Date(`${day}T00:00:00`);
  d.setDate(d.getDate() + n);
  return iso(d);
};

/** "A range of dates" in the New request builder (wireframes of 25 Oct): pick the dates, then the classes you teach
 * in them (all, all of a course, or some), then cancel them or move them.
 * - Cancel: one request, with or without make-up classes;
 * - Move: the teacher is away at those classes' times and the solver finds new ones; one request per week, as a
 *   request repairs one week.
 * Made with clicks, so no model reads it; the longest range is the coordinator's setting (Settings). */
export function RangeBuilder({ inst, cal, onSent }: { inst: InstanceView; cal: CalendarView; onSent: (id: string) => void }) {
  const { data: settings } = useApi(() => api.settings(), []);
  const longest = settings?.max_range_weeks ?? 4;
  const today = iso(new Date());
  const first = cal.weeks[0]?.monday ?? today;
  const last = cal.weeks.length ? plus(cal.weeks[cal.weeks.length - 1].monday, cal.days.length - 1) : today;
  const dateIn = (week: number, day: string) => plus(cal.weeks[week - 1].monday, cal.days.indexOf(day));
  const weekOf = (date: string) => cal.weeks.find((w) => w.monday <= date && date <= plus(w.monday, 6))?.week ?? null;

  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [from, setFrom] = useState(today > first ? today : first);
  const [to, setTo] = useState(() => plus(cal.weeks[startWeek(cal) - 1]?.monday ?? today, cal.days.length - 1));
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [action, setAction] = useState<Action>("move");
  const [makeup, setMakeup] = useState(true);
  const [reason, setReason] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const time = (s: number) => inst.slots.find((x) => x.slot === s)?.label ?? `slot ${s}`;
  const span = (weekOf(to) ?? 0) - (weekOf(from) ?? 0) + 1;
  const rangeError = to < from ? "The end is before the start." : span > longest ? `A range can cover at most ${longest} week${longest === 1 ? "" : "s"}.` : null;

  // the teacher's classes from the start to the end, not past and not already cancelled
  const items: Item[] = useMemo(
    () =>
      cal.weeks.flatMap((w) =>
        w.classes
          .filter((c) => !c.cancelled)
          .map((c) => ({ key: `${w.week}:${c.session}`, week: w.week, day: c.day, date: dateIn(w.week, c.day), slot: c.slot,
                         session: c.session, title: c.title, kind: c.kind, room: c.room }))
          .filter((x) => x.date >= from && x.date <= to && x.date >= today),
      ),
    [cal, from, to, today], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const chosen = items.filter((x) => picked.has(x.key));
  const courses = [...new Set(items.map((x) => x.title))];
  const byDate = items.reduce<Record<string, Item[]>>((acc, x) => ((acc[x.date] ??= []).push(x), acc), {});
  const toggle = (keys: string[], on: boolean) =>
    setPicked((p) => {
      const next = new Set(p);
      keys.forEach((k) => (on ? next.add(k) : next.delete(k)));
      return next;
    });
  const weeks = [...new Set(chosen.map((x) => x.week))].sort((a, b) => a - b);

  const quick = (start: number, count: number) => {
    const w = cal.weeks[start - 1];
    if (!w) return;
    setFrom(w.monday > today ? w.monday : today);
    setTo(plus(cal.weeks[Math.min(start + count - 1, cal.weeks.length) - 1].monday, cal.days.length - 1));
  };
  const now = startWeek(cal);

  const label = (x: Item) => `${x.title} ${x.kind} on ${dateOf(cal, x.week, x.day)} at ${time(x.slot)}`;
  const why = reason.trim() ? ` (${reason.trim().replace(/[.!?]$/, "")})` : "";
  const texts = (): { text: string; intents: Record<string, unknown>[] }[] => {
    const justification = reason.trim() ? "stated" : "none";
    if (action === "cancel") {
      const sessions = [...new Set(chosen.map((x) => x.session))];
      return [{
        text: `Please cancel my classes${why}: ${chosen.map(label).join("; ")}.${makeup ? " I will hold make-up classes." : ""}`,
        intents: sessions.map((s) => ({ kind: "cancel", which: { session: s }, justification,
                                         period: { weeks: [...new Set(chosen.filter((x) => x.session === s).map((x) => x.week))].sort((a, b) => a - b) } })),
      }];
    }
    return weeks.map((w) => {
      const these = chosen.filter((x) => x.week === w);
      return {
        text: `I am away${why}; please move my classes in week ${w}: ${these.map(label).join("; ")}.`,
        intents: these.map((x) => ({ kind: "unavailable", time: { days: [x.day], at: [clock(x.slot)] }, period: { weeks: [w] }, justification })),
      };
    });
  };

  const send = async () => {
    setSending(true);
    setError(null);
    let id: string | null = null;
    try {
      for (const r of texts()) id = (await api.submitRange(r.text, r.intents, action === "cancel" && makeup)).id;
      if (id) onSent(id);
    } catch (e) {
      setError((e as Error).message + (id ? ` (sent up to ${id})` : ""));
    } finally {
      setSending(false);
    }
  };

  const chip = (on: boolean, some = false) =>
    clsx("shrink-0 rounded-md border px-2.5 py-1 text-[12.5px] transition-colors",
         on ? "border-brand bg-brand/10 text-brand" : some ? "border-brand/50 text-ink" : "border-line text-ink-2 hover:bg-panel-2");
  if (step === 1)
    return (
      <div className="space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <DateField label="From" value={from} min={today > first ? today : first} max={last} onChange={setFrom} />
          <span className="pb-2 text-ink-3">→</span>
          <DateField label="To" value={to} min={from} max={last} onChange={setTo} />
        </div>
        <div className="flex flex-wrap gap-1.5">
          <span className="pt-1 text-[11.5px] font-medium uppercase tracking-wide text-ink-3">Quick</span>
          <button className={chip(false)} onClick={() => quick(now, 1)}>This week</button>
          {now < cal.weeks.length && <button className={chip(false)} onClick={() => quick(now + 1, 1)}>Next week</button>}
          {now < cal.weeks.length && <button className={chip(false)} onClick={() => quick(now, 2)}>Next 2 weeks</button>}
          {longest >= 4 && <button className={chip(false)} onClick={() => quick(now, Math.min(4, longest))}>Next 4 weeks</button>}
        </div>
        <div className="flex items-center justify-between gap-3">
          <p className={clsx("text-[12.5px]", rangeError ? "text-bad" : "text-ink-3")}>
            {rangeError ?? `${items.length} of your classes in these dates.`}
          </p>
          <Button variant="primary" size="sm" disabled={!!rangeError || !items.length} onClick={() => setStep(2)}>
            Show my classes →
          </Button>
        </div>
      </div>
    );

  const head = (n: 1 | 2, text: string) => (
    <button onClick={() => setStep(n)} className="flex items-center gap-1 text-[12.5px] text-ink-3 hover:text-brand">
      <ArrowLeft size={13} /> {text}
    </button>
  );
  const fmt = (d: string) => new Date(`${d}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short" });

  if (step === 2)
    return (
      <div className="space-y-3">
        {head(1, `${fmt(from)} – ${fmt(to)} · ${items.length} classes`)}
        <div className="flex flex-wrap items-center gap-1.5">
          <label className="mr-2 flex items-center gap-1.5 text-[13px]">
            <input type="checkbox" checked={!!items.length && chosen.length === items.length}
                   onChange={(e) => toggle(items.map((x) => x.key), e.target.checked)} />
            Select all
          </label>
          {courses.map((c) => {
            const keys = items.filter((x) => x.title === c).map((x) => x.key);
            const n = keys.filter((k) => picked.has(k)).length;
            return (
              <button key={c} className={chip(n === keys.length, n > 0)} onClick={() => toggle(keys, n !== keys.length)}
                      title={n === keys.length ? `Clear every ${c} class` : `Select every ${c} class`}>
                {n === keys.length ? "☑" : n ? "◐" : "☐"} {c} · {keys.length}
              </button>
            );
          })}
        </div>
        <div className="max-h-[45vh] space-y-3 overflow-y-auto pr-1">
          {Object.entries(byDate).map(([date, xs]) => (
            <div key={date}>
              <p className="mb-1 text-[11.5px] font-medium uppercase tracking-wide text-ink-3">{dateOf(cal, xs[0].week, xs[0].day)}</p>
              {xs.map((x) => (
                <label key={x.key} className="flex cursor-pointer items-center gap-2.5 rounded-md px-2 py-1.5 text-[13px] hover:bg-panel-2">
                  <input type="checkbox" checked={picked.has(x.key)} onChange={(e) => toggle([x.key], e.target.checked)} />
                  <span className="w-16 shrink-0 text-ink-3">{time(x.slot)}</span>
                  <span className="min-w-0 flex-1 truncate">{x.title} {x.kind}</span>
                  <span className="shrink-0 text-[12px] text-ink-3">{x.room}</span>
                </label>
              ))}
            </div>
          ))}
        </div>
        <div className="flex items-center justify-between gap-3 border-t border-line pt-3">
          <p className="text-[12.5px] text-ink-3">{chosen.length} selected</p>
          <Button variant="primary" size="sm" disabled={!chosen.length} onClick={() => setStep(3)}>
            Next: what to do
          </Button>
        </div>
      </div>
    );

  const out = texts();
  return (
    <div className="space-y-3">
      {head(2, `${chosen.length} class${chosen.length === 1 ? "" : "es"} · ${fmt(from)} – ${fmt(to)}`)}
      <div className="grid grid-cols-2 gap-2">
        {([["move", "Move them", CalendarClock], ["cancel", "Cancel them", XCircle]] as const).map(([k, text, Icon]) => (
          <button key={k} onClick={() => setAction(k)}
                  className={clsx("flex items-center justify-center gap-1.5 rounded-md border px-3 py-2 text-[13px] font-medium",
                                  action === k ? "border-brand bg-brand/10 text-brand" : "border-line hover:bg-panel-2")}>
            <Icon size={14} /> {text}
          </button>
        ))}
      </div>
      <p className="text-[12.5px] text-ink-3">
        {action === "move"
          ? `You are away at these times; the timetable office finds new times for each class in the same week.${weeks.length > 1 ? ` Sent as ${weeks.length} requests, one per week.` : ""}`
          : "The timetable coordinator arranges the cancellations."}
      </p>
      {action === "cancel" && (
        <label className="flex items-center gap-2 text-[13px]">
          <input type="checkbox" checked={makeup} onChange={(e) => setMakeup(e.target.checked)} /> Hold make-up classes later
        </label>
      )}
      <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="Reason (optional), e.g. conference in Delhi"
             className="h-8 w-full rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none placeholder:text-ink-3 focus:border-brand/60" />
      <div className="space-y-1.5 rounded-md bg-panel px-3 py-2">
        {out.map((r, i) => (
          <p key={i} className="text-[12.5px] text-ink-2">{r.text}</p>
        ))}
      </div>
      {error && <p className="text-[12.5px] text-bad">{error}</p>}
      <div className="flex justify-end">
        <Button variant="primary" size="sm" loading={sending} onClick={() => void send()}>
          {out.length > 1 ? `Send ${out.length} requests` : "Send request"}
        </Button>
      </div>
    </div>
  );
}

function DateField({ label, value, min, max, onChange }: { label: string; value: string; min: string; max: string; onChange: (v: string) => void }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11.5px] font-medium uppercase tracking-wide text-ink-3">{label}</span>
      <input type="date" value={value} min={min} max={max} onChange={(e) => e.target.value && onChange(e.target.value)}
             className="h-8 rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none [color-scheme:light_dark] focus:border-brand/60" />
    </label>
  );
}
