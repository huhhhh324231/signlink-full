import { spawn, spawnSync } from 'node:child_process';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(__dirname, '..');
const projectDir = path.resolve(webDir, '..');
const isWindows = process.platform === 'win32';
const apiScript = path.join(projectDir, 'vsl_web_api.py');
const apiHealthUrl = 'http://127.0.0.1:8008/api/vsl/health';

let apiProcess = null;
let viteProcess = null;

function candidatePythons() {
  const venvPython = path.join(projectDir, 'venv', 'Scripts', isWindows ? 'python.exe' : 'python');
  return [
    { command: venvPython, args: [] },
    { command: isWindows ? 'py.exe' : 'python3', args: isWindows ? ['-3'] : [] },
    { command: isWindows ? 'python.exe' : 'python', args: [] },
  ];
}

function findPython() {
  for (const candidate of candidatePythons()) {
    const result = spawnSync(candidate.command, [...candidate.args, '--version'], {
      cwd: projectDir,
      encoding: 'utf8',
      windowsHide: true,
    });

    if (result.status === 0) {
      const version = `${result.stdout}${result.stderr}`.trim();
      return { ...candidate, version };
    }
  }

  return null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function checkHealth() {
  return new Promise((resolve) => {
    const req = http.get(apiHealthUrl, { timeout: 1500 }, (res) => {
      let body = '';
      res.on('data', (chunk) => {
        body += chunk;
      });
      res.on('end', () => {
        try {
          const data = JSON.parse(body);
          resolve(Boolean(data.ok));
        } catch {
          resolve(false);
        }
      });
    });

    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
    req.on('error', () => resolve(false));
  });
}

async function waitForApi() {
  for (let attempt = 1; attempt <= 45; attempt += 1) {
    if (await checkHealth()) {
      return true;
    }
    await sleep(1000);
  }
  return false;
}

async function ensureApi() {
  if (await checkHealth()) {
    console.log('[api] Python API is already running at http://127.0.0.1:8008');
    return true;
  }

  console.log('[api] Starting Python API...');
  const python = findPython();
  if (!python) {
    console.error('[api] Could not find a working Python interpreter.');
    console.error('[api] Fix the project venv or install Python, then run: python -m venv venv && venv\\Scripts\\pip install -r requirements.txt');
    return false;
  }

  console.log(`[api] Using ${python.version}`);
  apiProcess = spawn(python.command, [...python.args, '-u', apiScript], {
    cwd: projectDir,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: false,
  });

  apiProcess.stdout.on('data', (data) => process.stdout.write(`[api] ${data}`));
  apiProcess.stderr.on('data', (data) => process.stderr.write(`[api] ${data}`));
  apiProcess.on('exit', (code) => {
    if (code !== 0 && code !== null) {
      console.error(`[api] Python API exited with code ${code}`);
    }
  });

  return waitForApi();
}

function startVite() {
  const viteCommand = path.join(webDir, 'node_modules', '.bin', isWindows ? 'vite.cmd' : 'vite');
  viteProcess = spawn(viteCommand, ['--port=3000', '--host=0.0.0.0'], {
    cwd: webDir,
    stdio: 'inherit',
    shell: isWindows,
    windowsHide: false,
  });

  viteProcess.on('exit', (code) => {
    cleanup();
    process.exit(code ?? 0);
  });
}

function cleanup() {
  if (viteProcess && !viteProcess.killed) {
    viteProcess.kill();
  }
  if (apiProcess && !apiProcess.killed) {
    apiProcess.kill();
  }
}

process.on('SIGINT', () => {
  cleanup();
  process.exit(0);
});
process.on('SIGTERM', () => {
  cleanup();
  process.exit(0);
});

const ready = await ensureApi();
if (!ready) {
  console.error('[api] Python API did not become ready. Check vsl_web_api.py output above.');
  cleanup();
  process.exit(1);
}

startVite();
