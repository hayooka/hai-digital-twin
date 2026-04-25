import { createClient, SupabaseClient } from '@supabase/supabase-js';

export interface Review {
  id: string;
  name: string;
  role: string;
  rating: number;
  text: string;
  created_at: string;
}

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

let cached: SupabaseClient | null = null;

function getClient(): SupabaseClient {
  if (cached) return cached;
  if (!supabaseUrl || !supabaseAnonKey) {
    // Build/SSR with no env vars — return an empty stub so module load succeeds.
    // The Reviews component will see calls fail and render "no reviews yet".
    throw new Error(
      'Supabase env vars (NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY) ' +
      'are not set. Reviews submission/listing will not work until they are added in ' +
      'Netlify → Site settings → Environment variables.'
    );
  }
  cached = createClient(supabaseUrl, supabaseAnonKey);
  return cached;
}

// Proxy: createClient is NOT called at module load. Only on first attribute
// access (e.g. supabase.from(...)). Build-time prerender doesn't access it.
export const supabase = new Proxy({} as SupabaseClient, {
  get(_t, prop) {
    return (getClient() as any)[prop];
  },
});
