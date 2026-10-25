import { useEffect, useState } from "react";
import { Button, Card, PageHeader, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";

/** What the coordinator sets for everyone's requests. For now: the longest range of dates a request may cover
 * (the range builder, "A range of dates"). */
export default function Settings() {
  const { data, refresh } = useApi(() => api.settings(), []);
  const [weeks, setWeeks] = useState("");
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  useEffect(() => {
    if (data) setWeeks(String(data.max_range_weeks));
  }, [data]);

  const n = Number(weeks);
  const valid = Number.isInteger(n) && n >= 1 && n <= 52;
  const save = async () => {
    setSaving(true);
    setNote(null);
    try {
      await api.setSettings(n);
      await refresh();
      setNote("Saved.");
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <PageHeader title="Settings" subtitle="What applies to everyone's requests." />
      <Card className="max-w-xl p-4">
        <p className="text-[13.5px] font-medium">Longest range of dates in a request</p>
        <p className="mt-0.5 text-[12.5px] text-ink-3">
          Teachers can cancel or move several classes of a range of dates in one go. A range can cover at most this many weeks.
        </p>
        {!data ? (
          <Skeleton className="mt-3 h-8 w-48" />
        ) : (
          <div className="mt-3 flex items-center gap-2">
            <input type="number" min={1} max={52} value={weeks} onChange={(e) => setWeeks(e.target.value)}
                   className="h-8 w-20 rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none focus:border-brand/60" />
            <span className="text-[13px] text-ink-2">weeks</span>
            <Button size="sm" variant="primary" loading={saving} disabled={!valid || n === data.max_range_weeks} onClick={() => void save()}>
              Save
            </Button>
            {note && <span className="text-[12.5px] text-ink-3">{note}</span>}
          </div>
        )}
      </Card>
    </div>
  );
}
