package vecgo

import (
	"context"
	"log/slog"
	"os"
)

// Logger wraps slog.Logger with vecgo-specific context.
// This provides structured logging with consistent field names.
type Logger struct {
	*slog.Logger
}

// NewLogger creates a new Logger with the given handler.
// If handler is nil, uses default text handler to stderr.
func NewLogger(handler slog.Handler) *Logger {
	if handler == nil {
		handler = slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		})
	}
	return &Logger{
		Logger: slog.New(handler),
	}
}

// NewJSONLogger creates a Logger that outputs JSON-formatted logs.
// level sets the minimum log level (e.g., slog.LevelDebug, slog.LevelInfo).
func NewJSONLogger(level slog.Level) *Logger {
	handler := slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{
		Level: level,
	})
	return &Logger{
		Logger: slog.New(handler),
	}
}

// NewTextLogger creates a Logger that outputs human-readable text logs.
func NewTextLogger(level slog.Level) *Logger {
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: level,
	})
	return &Logger{
		Logger: slog.New(handler),
	}
}

// NoopLogger creates a Logger that discards all log output.
// Use this to disable logging entirely.
func NoopLogger() *Logger {
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.Level(1000), // Unreachable level
	})
	return &Logger{
		Logger: slog.New(handler),
	}
}

// WithContext adds context values to the logger.
func (l *Logger) WithContext(ctx context.Context) *Logger {
	return &Logger{
		Logger: l.Logger.With(),
	}
}

// WithID adds an ID field to the logger (useful for tagging operations).
func (l *Logger) WithID(id uint64) *Logger {
	return &Logger{
		Logger: l.Logger.With("id", id),
	}
}

// WithK adds a k (neighbor count) field to the logger.
func (l *Logger) WithK(k int) *Logger {
	return &Logger{
		Logger: l.Logger.With("k", k),
	}
}

// WithDimension adds a dimension field to the logger.
func (l *Logger) WithDimension(dim int) *Logger {
	return &Logger{
		Logger: l.Logger.With("dimension", dim),
	}
}

// WithCount adds a count field to the logger.
func (l *Logger) WithCount(count int) *Logger {
	return &Logger{
		Logger: l.Logger.With("count", count),
	}
}

// LogInsert logs an insert operation.
func (l *Logger) LogInsert(ctx context.Context, id uint64, dimension int, err error) {
	if err != nil {
		l.ErrorContext(ctx, "insert failed",
			"id", id,
			"dimension", dimension,
			"error", err,
		)
	} else {
		l.DebugContext(ctx, "insert completed",
			"id", id,
			"dimension", dimension,
		)
	}
}

// LogBatchInsert logs a batch insert operation.
func (l *Logger) LogBatchInsert(ctx context.Context, count, failed int) {
	if failed > 0 {
		l.WarnContext(ctx, "batch insert completed with failures",
			"total", count,
			"failed", failed,
			"success", count-failed,
		)
	} else {
		l.InfoContext(ctx, "batch insert completed",
			"count", count,
		)
	}
}

// LogSearch logs a search operation.
func (l *Logger) LogSearch(ctx context.Context, k, resultsFound int, err error) {
	if err != nil {
		l.ErrorContext(ctx, "search failed",
			"k", k,
			"error", err,
		)
	} else {
		l.DebugContext(ctx, "search completed",
			"k", k,
			"results", resultsFound,
		)
	}
}

// LogDelete logs a delete operation.
func (l *Logger) LogDelete(ctx context.Context, id uint64, err error) {
	if err != nil {
		l.ErrorContext(ctx, "delete failed",
			"id", id,
			"error", err,
		)
	} else {
		l.DebugContext(ctx, "delete completed",
			"id", id,
		)
	}
}

// LogUpdate logs an update operation.
func (l *Logger) LogUpdate(ctx context.Context, id uint64, err error) {
	if err != nil {
		l.ErrorContext(ctx, "update failed",
			"id", id,
			"error", err,
		)
	} else {
		l.DebugContext(ctx, "update completed",
			"id", id,
		)
	}
}

// LogSnapshot logs a snapshot operation.
func (l *Logger) LogSnapshot(ctx context.Context, filename string, err error) {
	if err != nil {
		l.ErrorContext(ctx, "snapshot failed",
			"filename", filename,
			"error", err,
		)
	} else {
		l.InfoContext(ctx, "snapshot saved",
			"filename", filename,
		)
	}
}

// LogRecovery logs a WAL recovery operation.
func (l *Logger) LogRecovery(ctx context.Context, entriesReplayed int, err error) {
	if err != nil {
		l.ErrorContext(ctx, "WAL recovery failed",
			"entries_replayed", entriesReplayed,
			"error", err,
		)
	} else {
		l.InfoContext(ctx, "WAL recovery completed",
			"entries_replayed", entriesReplayed,
		)
	}
}
