create extension if not exists pgcrypto;

create table if not exists public.analysis_runs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  dataset_name text,
  target_col text not null,
  horizon integer,
  test_size integer,
  best_model text,
  best_metrics jsonb,
  model_metrics jsonb,
  anomalies jsonb,
  report_quality jsonb,
  knowledge_context text,
  report text
);

alter table public.analysis_runs enable row level security;

drop policy if exists "允许公开读取分析记录" on public.analysis_runs;
create policy "允许公开读取分析记录"
on public.analysis_runs for select
to anon
using (true);

drop policy if exists "允许公开写入分析记录" on public.analysis_runs;
create policy "允许公开写入分析记录"
on public.analysis_runs for insert
to anon
with check (true);
