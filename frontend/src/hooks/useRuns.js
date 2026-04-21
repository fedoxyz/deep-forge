import { useState, useEffect, useCallback } from 'react';

export function useRuns() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchRuns = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/training/runs');
      const data = await res.json();
      setRuns(data.runs || []);
    } catch {}
    setLoading(false);
  }, []);

  useEffect(() => { fetchRuns(); }, [fetchRuns]);

  return { runs, loading, refresh: fetchRuns };
}
