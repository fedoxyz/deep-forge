import { useState, useEffect, useRef, useCallback } from 'react';
import { getTrainingStatus, subscribeTraining } from '../utils/api';

export function useTrainingStatus(pollInterval = 2000) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  const poll = useCallback(async () => {
    try {
      const data = await getTrainingStatus();
      setStatus(data);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, pollInterval);
    return () => clearInterval(id);
  }, [poll, pollInterval]);

  return { status, error, refresh: poll };
}

export function useTrainingStream() {
  const [data, setData] = useState(null);
  const unsubRef = useRef(null);

  const start = useCallback(() => {
    if (unsubRef.current) unsubRef.current();
    unsubRef.current = subscribeTraining(setData);
  }, []);

  const stop = useCallback(() => {
    if (unsubRef.current) { unsubRef.current(); unsubRef.current = null; }
  }, []);

  useEffect(() => () => stop(), [stop]);

  return { data, start, stop };
}
