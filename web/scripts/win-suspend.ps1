# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2019-2026 Jean-David Moisan
#
# Suspend or resume a whole process tree on Windows, by PID. Used by loop-ctl.mjs to
# freeze the train:loop (and the native gen/match/refresh children it spawns) so a long
# run can be paused mid-step to free the CPU, then thawed exactly where it left off.
# Suspended threads consume no CPU and keep all in-memory state, so no work is lost.
#
#   pwsh -File win-suspend.ps1 -RootPid <pid> -Action suspend|resume
#
# Prints the number of processes affected. Uses ntdll NtSuspendProcess/NtResumeProcess
# (freezes every thread of a process atomically), walked over the descendant tree.
param(
  [Parameter(Mandatory = $true)][int]$RootPid,
  [Parameter(Mandatory = $true)][ValidateSet('suspend', 'resume')][string]$Action
)
$ErrorActionPreference = 'Stop'

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class ProcCtl {
  [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr OpenProcess(int access, bool inherit, int pid);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool CloseHandle(IntPtr h);
  [DllImport("ntdll.dll")] static extern uint NtSuspendProcess(IntPtr h);
  [DllImport("ntdll.dll")] static extern uint NtResumeProcess(IntPtr h);
  const int PROCESS_SUSPEND_RESUME = 0x0800;
  public static void Suspend(int pid){ var h=OpenProcess(PROCESS_SUSPEND_RESUME,false,pid); if(h!=IntPtr.Zero){ NtSuspendProcess(h); CloseHandle(h);} }
  public static void Resume(int pid){ var h=OpenProcess(PROCESS_SUSPEND_RESUME,false,pid); if(h!=IntPtr.Zero){ NtResumeProcess(h); CloseHandle(h);} }
}
"@

# Build a parent -> children map of every live process, then BFS from the root.
$children = @{}
foreach ($p in Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId) {
  $ppid = [int]$p.ParentProcessId
  if (-not $children.ContainsKey($ppid)) { $children[$ppid] = New-Object System.Collections.Generic.List[int] }
  $children[$ppid].Add([int]$p.ProcessId)
}
$tree = New-Object System.Collections.Generic.List[int]
$seen = @{}
$queue = New-Object System.Collections.Generic.Queue[int]
$queue.Enqueue($RootPid)
while ($queue.Count -gt 0) {
  $cur = $queue.Dequeue()
  if ($seen.ContainsKey($cur)) { continue }
  $seen[$cur] = $true
  $tree.Add($cur)
  if ($children.ContainsKey($cur)) { foreach ($c in $children[$cur]) { $queue.Enqueue($c) } }
}

# Suspend parent-before-child (root first) so a parent can't spawn a new child after we've
# already passed it; resume in the reverse order. ($pid is an automatic variable — use $t.)
if ($Action -eq 'resume') { $tree.Reverse() }
foreach ($t in $tree) {
  if ($Action -eq 'suspend') { [ProcCtl]::Suspend($t) } else { [ProcCtl]::Resume($t) }
}
Write-Output $tree.Count
