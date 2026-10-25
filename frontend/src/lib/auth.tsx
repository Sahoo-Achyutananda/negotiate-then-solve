import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, currentUserId, setCurrentUserId } from "./api";
import type { Persona, Role, View } from "./types";

interface Auth {
  user: Persona | null;
  ready: boolean;
  signIn: (id: string) => Promise<void>;
  signOut: () => void;
  can: (what: Capability) => boolean;
  /** Whether this person's role has the view (the server's list). */
  sees: (view: View) => boolean;
}

export type Capability = "approve" | "see_all" | "observe" | "request" | "negotiate";

const CAPS: Record<Capability, string[]> = {
  approve: ["coordinator"],
  see_all: ["coordinator", "hod", "dean"],
  observe: ["coordinator", "hod", "dean"],
  request: ["faculty", "hod", "guest_faculty", "lab_incharge", "student", "exam_cell", "coordinator"],
  negotiate: ["faculty", "hod", "guest_faculty"],
};

/** Mirror of ``VIEWS`` in nts/web.py, used only when an older server sends no list. */
const TEACHING: Role[] = ["hod", "faculty", "guest_faculty"];
const FALLBACK_VIEWS: Record<View, Role[]> = {
  home: [], requests: [], timetable: [], handbook: [], how: [], // everyone
  new: ["coordinator", "hod", "faculty", "guest_faculty", "lab_incharge", "exam_cell", "student"],
  inbox: [],
  approvals: ["coordinator"],
  history: ["coordinator"],
  documents: ["coordinator"],
  semester: ["coordinator"],
  prefs: ["coordinator", "hod", "faculty", "guest_faculty", "student"],
  clubs: ["coordinator", "student"],
  fairness: ["coordinator", "dean", ...TEACHING],
  graph: ["coordinator"],
  health: ["coordinator"],
  experiments: ["coordinator"],
  study: ["coordinator"],
  settings: ["coordinator"],
};

const AuthContext = createContext<Auth | null>(null);

/** Mock sign-in: the backend trusts the X-User header. Swap for the college
 * OIDC flow; the server already makes every authorisation decision. */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<Persona | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    // Mock-auth convenience for demos: ?as=F-101 signs in as that person.
    const params = new URLSearchParams(window.location.search);
    const as = params.get("as");
    if (as) {
      params.delete("as");
      const rest = params.toString();
      window.history.replaceState(null, "", window.location.pathname + (rest ? `?${rest}` : ""));
      api
        .login(as)
        .then((p) => {
          setCurrentUserId(p.id);
          setUser(p);
        })
        .catch(() => setCurrentUserId(null))
        .finally(() => setReady(true));
      return;
    }
    if (!currentUserId()) {
      setReady(true);
      return;
    }
    api
      .me()
      .then(setUser)
      .catch(() => setCurrentUserId(null))
      .finally(() => setReady(true));
  }, []);

  const signIn = useCallback(async (id: string) => {
    const p = await api.login(id);
    setCurrentUserId(p.id);
    setUser(p);
  }, []);

  const signOut = useCallback(() => {
    setCurrentUserId(null);
    setUser(null);
  }, []);

  const can = useCallback((what: Capability) => !!user && CAPS[what].includes(user.role), [user]);
  const sees = useCallback(
    (view: View) => {
      if (!user) return false;
      if (user.views) return user.views.includes(view);
      const roles = FALLBACK_VIEWS[view];
      return !roles.length || roles.includes(user.role);
    },
    [user],
  );

  const value = useMemo(() => ({ user, ready, signIn, signOut, can, sees }), [user, ready, signIn, signOut, can, sees]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): Auth {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth outside AuthProvider");
  return ctx;
}
