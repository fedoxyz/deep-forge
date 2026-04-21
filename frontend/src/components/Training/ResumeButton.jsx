import React from 'react';
import { RotateCcw } from 'lucide-react';
import { resumeTraining } from '../../utils/api';

export default function ResumeButton({ status, selectedConfig, mode, setMsg, setLoading, refresh }) {
  const canResume = (status?.status === 'completed' || status?.status === 'error') && status?.run_dir;

  if (!canResume) return null;

  const handleResume = async () => {
    if (!selectedConfig) { setMsg('Select a config first'); return; }
    setLoading(true);
    try {
      await resumeTraining(selectedConfig, mode);
      setMsg('Resuming training...');
      refresh();
    } catch (e) { setMsg('Error: ' + e.message); }
    setLoading(false);
  };

  return (
    <button
      onClick={handleResume}
      className="flex items-center gap-2 px-6 py-2.5 bg-forge-accent text-black rounded-lg text-sm font-semibold hover:bg-forge-accent/90"
    >
      <RotateCcw className="w-4 h-4" /> Resume
    </button>
  );
}
