# Failure Recovery and Skiplist

One of the most important operational realities of GlossAPI is that Docling can fail in ways that matter at corpus scale.

This document describes the recovery model that exists to keep runs usable when that happens.

## Problem statement

Docling can provide excellent throughput, but under some conditions it may:

- crash on specific PDFs
- hang or time out
- fail more often when batch size is greater than 1
- make a worker or runner unusable after a bad document

At corpus scale, this is not a corner case. It is a design constraint.

## Recovery philosophy

GlossAPI does not assume every document is processable in a single clean pass.

Instead, the recovery model aims to:

1. preserve successful work
2. isolate problematic documents
3. let the run continue where possible
4. make the failure visible in retained metadata and artifacts
5. avoid repeatedly crashing on the same bad inputs

## What the skiplist is for

The skiplist exists to record documents that should not be retried blindly during the same operational flow.

Its role is practical:

- prevent repeated crashes on known-problematic files
- preserve throughput for the rest of the corpus
- make failure handling explicit rather than implicit

The skiplist is therefore part of the recovery system, not a convenience feature.

## Runner recovery

When the extraction or OCR path encounters problematic PDFs, the runner may need to:

- discard the broken processing context
- reinitialize the relevant engine
- continue with the remaining documents

The important requirement is that recovery should preserve the ability to reason about the run afterward.

That means the pipeline should leave behind enough evidence to answer:

- which file caused the problem
- which stage failed
- whether the document was skipped, timed out, or crashed the worker
- whether the rest of the run continued successfully

## Relationship to batching

Batching is strongly connected to failure recovery.

With batch size greater than 1, one bad document can affect:

- other documents in the same batch
- the current worker state
- the confidence that outputs from the current runner remain valid

That is why recovery semantics are just as important as batching semantics.

## Expected operator understanding

Operators should know:

- skiplist use is not a silent corruption mechanism
- skipped files are expected to be inspectable later
- keeping the rest of the run alive is often better than insisting on full completion in one pass
- repeated crashes on the same file should feed into the skiplist instead of consuming the whole run

## Expected contributor understanding

Contributors should know:

- any new batching or concurrency logic must preserve recoverability
- a failure on one file should not erase already successful outputs
- stage metadata should remain analyzable after recovery
- skip behavior should be explicit in metadata and logs

## Documentation checklist for this area

Any future change in this area should update docs if it affects:

- skiplist file location
- semantics of a skipped document
- crash vs timeout classification
- rerun behavior after recovery
- whether a failed document can poison a worker or batch
- which artifacts remain authoritative after a partial stage failure

## Long-term architectural note

The skiplist solves an operational problem in the current architecture. It should not be mistaken for the ideal long-term abstraction.

Long-term improvements may include:

- more explicit per-document status records
- better process isolation
- chunk-aware scheduling
- less dependence on filename-only identity

But in the current system, the skiplist is a key part of practical reliability.
