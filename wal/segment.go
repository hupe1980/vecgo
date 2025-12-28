package wal

// This file exists to match the refactor proposal layout.
//
// A future refactor may split the WAL into segments (e.g. wal-000001.log) to
// improve truncation and corruption isolation. The current WAL implementation
// uses a single file (vecgo.wal) with an explicit checkpoint marker.
