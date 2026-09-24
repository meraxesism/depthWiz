#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import net from 'node:net';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = __dirname;
const uiPort = 5173;
const tsApiPort = 8787;
const pythonApiPort = 8788;
const venvDir = path.join(projectRoot, '.venv');
const requirementsFile = path.join(projectRoot, 'requirements-backend.txt');

function hasFile(filePath) {
  try {
    fs.accessSync(filePath, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

function findFiles(rootDir, predicate, maxDepth = 5) {
  const matches = [];

  function walk(currentDir, depth) {
    if (depth > maxDepth) return;
    let entries;
    try {
      entries = fs.readdirSync(currentDir, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath, depth + 1);
        continue;
      }
      if (predicate(entry.name, fullPath)) {
        matches.push(fullPath);
      }
    }
  }

  walk(rootDir, 0);
  return matches;
}

function getVenvPython() {
  if (process.platform === 'win32') {
    return path.join(venvDir, 'Scripts', 'python.exe');
  }
  return path.join(venvDir, 'bin', 'python');
}

function resolvePython() {
  const explicit = process.env.DEPTHWIZ_PYTHON || process.env.PYTHON || process.env.PYTHON_PATH;
  if (explicit) {
    if (hasFile(explicit)) return explicit;
    const venvPython = getVenvPython();
    if (hasFile(venvPython)) return venvPython;
  }

  const venvPython = getVenvPython();
  if (hasFile(venvPython)) return venvPython;

  const candidates = [];
  if (process.platform === 'win32') {
    candidates.push(path.join(projectRoot, '..', '.venv', 'Scripts', 'python.exe'));
  } else {
    candidates.push(path.join(projectRoot, '..', '.venv', 'bin', 'python'));
  }

  for (const candidate of candidates) {
    if (hasFile(candidate)) return candidate;
  }

  for (const candidate of ['python3', 'python']) {
    const result = spawnSync(candidate, ['--version'], { stdio: 'ignore' });
    if (result.status === 0) return candidate;
  }

  return null;
}

function ensureNodeDependencies() {
  const nodeModules = path.join(projectRoot, 'node_modules');
  if (hasFile(nodeModules)) return;

  const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  console.log('Installing Node.js dependencies...');
  const result = spawnSync(npmCommand, ['install'], {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });

  if (result.status !== 0) {
    throw new Error('npm install failed');
  }
}

function ensurePythonVenv() {
  const venvPython = getVenvPython();
  if (hasFile(venvPython)) return venvPython;

  const basePython = resolvePython();
  if (!basePython) {
    throw new Error('Python not found. Install Python and try again or set DEPTHWIZ_PYTHON.');
  }

  console.log('Creating Python virtual environment in .venv...');
  const result = spawnSync(basePython, ['-m', 'venv', '.venv'], {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });

  if (result.status !== 0) {
    throw new Error('Failed to create the Python virtual environment');
  }

  return venvPython;
}

function ensurePythonDependencies(pythonExecutable) {
  if (!hasFile(requirementsFile)) {
    throw new Error(`Python requirements file not found: ${requirementsFile}`);
  }

  console.log('Installing Python backend dependencies...');
  const upgrade = spawnSync(pythonExecutable, ['-m', 'pip', 'install', '--upgrade', 'pip'], {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });

  if (upgrade.status !== 0) {
    throw new Error('Failed to upgrade pip in the local virtual environment');
  }

  const install = spawnSync(pythonExecutable, ['-m', 'pip', 'install', '-r', 'requirements-backend.txt'], {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });

  if (install.status !== 0) {
    throw new Error('Failed to install backend Python dependencies');
  }
}

function resolveModelPath() {
  const explicit = process.env.DEPTHWIZ_DEPTH_MODEL_PATH;
  if (explicit && hasFile(explicit)) return explicit;

  const candidates = [
    'depth-anything-v2-small-quantized.onnx',
    'depth-anything-v2-small/model_quantized.onnx',
    'models/depth-anything-v2-small/model_quantized.onnx',
    'model_quantized.onnx',
  ];

  for (const candidate of candidates) {
    const resolved = path.join(projectRoot, candidate);
    if (hasFile(resolved)) return resolved;
  }

  const matches = findFiles(projectRoot, (name) => /(?:depth.*anything|model_?quantized|quantized).*\.onnx$/i.test(name));
  if (matches.length > 0) return matches[0];

  return null;
}

function isPortInUse(port) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    socket.setTimeout(500);

    socket.on('connect', () => {
      socket.destroy();
      resolve(true);
    });

    socket.on('timeout', () => {
      socket.destroy();
      resolve(false);
    });

    socket.on('error', () => {
      resolve(false);
    });

    socket.connect(port, '127.0.0.1');
  });
}

function spawnCommand(label, command, args, options) {
  const child = spawn(command, args, {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: process.platform === 'win32',
    env: { ...process.env, ...options.env },
  });

  child.on('exit', (code) => {
    if (code !== 0) {
      console.log(`\n${label} exited with code ${code}.`);
    }
  });

  return child;
}

async function main() {
  const args = process.argv.slice(2);
  const isDryRun = args.includes('--dry-run') || args.includes('--check');
  const isBootstrapOnly = args.includes('--bootstrap-only') || args.includes('--setup');

  if (args.includes('--help') || args.includes('-h')) {
    console.log('Usage: node start-depthwiz.js [--dry-run] [--bootstrap-only]');
    console.log('');
    console.log('Starts the DepthWizard TypeScript API, Vite frontend, and Python API if ports are free.');
    console.log('Environment overrides: DEPTHWIZ_PYTHON, DEPTHWIZ_DEPTH_MODEL_PATH');
    process.exit(0);
  }

  const python = resolvePython();
  const modelPath = resolveModelPath();

  console.log('DepthWizard launcher');
  console.log(`Project root: ${projectRoot}`);
  console.log(`Python: ${python || 'not found'}`);
  console.log(`Model: ${modelPath || 'not found'}`);

  if (isDryRun) {
    process.exit(0);
  }

  if (!python) {
    console.error('ERROR: Python was not found on PATH and no local .venv Python executable was detected.');
    console.error('Create a virtual environment or set DEPTHWIZ_PYTHON to the Python executable you want to use.');
    process.exit(1);
  }

  ensureNodeDependencies();

  const venvPython = ensurePythonVenv();
  ensurePythonDependencies(venvPython);

  if (isBootstrapOnly) {
    console.log('');
    console.log(`Python environment ready at ${venvDir}`);
    console.log(`Use ${venvPython} to run backend commands directly.`);
    process.exit(0);
  }

  const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const envForPython = {
    ...process.env,
    DEPTHWIZ_DEPTH_MODEL_PATH: modelPath || process.env.DEPTHWIZ_DEPTH_MODEL_PATH || '',
  };

  const launchPlan = [
    {
      label: 'TypeScript API',
      checkPort: tsApiPort,
      command: npmCommand,
      args: ['run', 'server'],
      env: {},
    },
    {
      label: 'Vite frontend',
      checkPort: uiPort,
      command: npmCommand,
      args: ['run', 'dev'],
      env: {},
    },
    {
      label: 'Python API',
      checkPort: pythonApiPort,
      command: venvPython,
      args: ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8788'],
      env: envForPython,
    },
  ];

  for (const service of launchPlan) {
    if (await isPortInUse(service.checkPort)) {
      console.log(`${service.label} already running on port ${service.checkPort}; skipping launch.`);
      continue;
    }

    console.log(`Starting ${service.label} on port ${service.checkPort}...`);
    spawnCommand(service.label, service.command, service.args, { env: service.env });
  }

  console.log('');
  console.log('Frontend:    http://localhost:5173/');
  console.log('TypeScript:  http://localhost:8787/api/health');
  console.log('Python API:  http://localhost:8788/health');
  console.log('');
  console.log('Use Ctrl+C in each terminal to stop the services.');
}

main().catch((error) => {
  console.error('Failed to start DepthWizard:', error);
  process.exit(1);
});
