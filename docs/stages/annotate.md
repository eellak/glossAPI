# Annotate Stage

## Purpose

The annotate stage classifies extracted sections and optionally produces more complete annotation outputs.

## Main responsibilities

- load section parquet outputs
- apply the packaged section-classifier model
- optionally perform fuller annotation behavior
- preserve document-type context when available

## Main inputs

- section parquet outputs
- classifier model
- optional document-type metadata

## Main outputs

- classified section parquet
- fully annotated section parquet when enabled

## Dependency note

This stage depends on model availability. If the model is missing, the stage should fail clearly or skip clearly rather than silently pretending annotation succeeded.

## Metadata role

This stage can enrich downstream outputs with document-type and processing-stage information.

## Contributor note

Changes here should preserve compatibility with the section parquet contract and should document any new output schema or annotation semantics.
