/**
 * Lightweight console log capture for the debug report.
 *
 * Wraps console.log/info/warn/error and stores the last N entries in a ring
 * buffer. The original console behavior is preserved — every call still goes
 * to the dev tools.
 */

export interface CapturedLog {
  level: 'log' | 'info' | 'warn' | 'error';
  timestamp: number;
  message: string;
}

const MAX_LOGS = 200;
const buffer: CapturedLog[] = [];

function safeStringify(value: unknown): string {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (value instanceof Error) {
    return `${value.name}: ${value.message}${value.stack ? '\n' + value.stack : ''}`;
  }
  try {
    return JSON.stringify(value, (_key, val) => {
      if (val instanceof File) return `[File ${val.name} ${val.size}B ${val.type}]`;
      if (val instanceof Blob) return `[Blob ${val.size}B ${val.type}]`;
      if (val instanceof HTMLElement) return `[HTMLElement ${val.tagName}]`;
      if (val && typeof val === 'object' && val.constructor && val.constructor.name !== 'Object' && val.constructor.name !== 'Array') {
        return `[${val.constructor.name}]`;
      }
      return val;
    });
  } catch {
    return String(value);
  }
}

function record(level: CapturedLog['level'], args: unknown[]) {
  const message = args.map(safeStringify).join(' ');
  buffer.push({ level, timestamp: Date.now(), message });
  if (buffer.length > MAX_LOGS) buffer.shift();
}

let installed = false;
export function installLogCapture(): void {
  if (installed) return;
  installed = true;

  const original = {
    log: console.log.bind(console),
    info: console.info.bind(console),
    warn: console.warn.bind(console),
    error: console.error.bind(console),
  };

  console.log = (...args: unknown[]) => {
    record('log', args);
    original.log(...args);
  };
  console.info = (...args: unknown[]) => {
    record('info', args);
    original.info(...args);
  };
  console.warn = (...args: unknown[]) => {
    record('warn', args);
    original.warn(...args);
  };
  console.error = (...args: unknown[]) => {
    record('error', args);
    original.error(...args);
  };

  window.addEventListener('error', (e) => {
    record('error', [`[uncaught] ${e.message} @ ${e.filename}:${e.lineno}:${e.colno}`]);
  });
  window.addEventListener('unhandledrejection', (e) => {
    record('error', [`[unhandledrejection] ${safeStringify(e.reason)}`]);
  });
}

export function getCapturedLogs(): CapturedLog[] {
  return buffer.slice();
}

export function clearCapturedLogs(): void {
  buffer.length = 0;
}
