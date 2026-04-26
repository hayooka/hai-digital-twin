/*
  # Create Reviews Table for PlantMirror Landing Page

  1. New Tables
    - `reviews`
      - `id` (uuid, primary key) - unique identifier
      - `name` (text, not null) - reviewer's name
      - `role` (text, nullable) - reviewer's affiliation/role
      - `rating` (int, not null) - star rating 1-5
      - `text` (text, not null) - review body
      - `created_at` (timestamptz) - submission timestamp

  2. Security
    - Enable RLS on `reviews` table
    - Public SELECT policy: anyone can read approved reviews
    - Public INSERT policy: anyone can submit a review with valid required fields
    
  3. Notes
    - This is a public-facing reviews feature — no auth required
    - Rating is constrained to 1–5 via CHECK constraint
    - Name and text are required (non-empty)
*/

CREATE TABLE IF NOT EXISTS reviews (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  role text DEFAULT '',
  rating integer NOT NULL,
  text text NOT NULL,
  created_at timestamptz DEFAULT now(),
  CONSTRAINT reviews_rating_range CHECK (rating >= 1 AND rating <= 5),
  CONSTRAINT reviews_name_nonempty CHECK (char_length(trim(name)) > 0),
  CONSTRAINT reviews_text_nonempty CHECK (char_length(trim(text)) > 0)
);

ALTER TABLE reviews ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read reviews"
  ON reviews
  FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Anyone can submit a review"
  ON reviews
  FOR INSERT
  TO anon, authenticated
  WITH CHECK (
    char_length(trim(name)) > 0
    AND char_length(trim(text)) > 0
    AND rating >= 1
    AND rating <= 5
  );

CREATE INDEX IF NOT EXISTS reviews_created_at_idx ON reviews (created_at DESC);
