'use client';

import { useState, useEffect, useCallback } from 'react';
import { supabase, type Review } from '@/lib/supabase';

function StarRating({
  value,
  onChange,
  readOnly = false,
  size = 'md',
}: {
  value: number;
  onChange?: (v: number) => void;
  readOnly?: boolean;
  size?: 'sm' | 'md';
}) {
  const [hovered, setHovered] = useState(0);
  const sz = size === 'sm' ? 'w-4 h-4' : 'w-6 h-6';

  return (
    <div
      className="flex items-center gap-0.5"
      role={readOnly ? 'img' : 'group'}
      aria-label={`${value} out of 5 stars`}
    >
      {[1, 2, 3, 4, 5].map((star) => {
        const filled = (hovered || value) >= star;
        return (
          <button
            key={star}
            type="button"
            disabled={readOnly}
            onClick={() => onChange?.(star)}
            onMouseEnter={() => !readOnly && setHovered(star)}
            onMouseLeave={() => !readOnly && setHovered(0)}
            aria-label={readOnly ? undefined : `Rate ${star} star${star > 1 ? 's' : ''}`}
            className={`transition-transform duration-100 ${
              readOnly ? 'cursor-default' : 'cursor-pointer hover:scale-110'
            }`}
          >
            <svg
              viewBox="0 0 24 24"
              className={`${sz} transition-colors duration-100 ${
                filled ? 'text-amber-400 fill-amber-400' : 'text-slate-600 fill-slate-600'
              }`}
              aria-hidden="true"
            >
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
            </svg>
          </button>
        );
      })}
    </div>
  );
}

function ReviewCard({ review }: { review: Review }) {
  const date = new Date(review.created_at).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });

  return (
    <div className="card-base rounded-xl p-5 border border-[#1e2d45]">
      <div className="flex items-start justify-between gap-4 mb-3">
        <div>
          <div className="text-white font-medium text-sm">{review.name}</div>
          {review.role && (
            <div className="text-slate-500 text-xs mt-0.5">{review.role}</div>
          )}
        </div>
        <div className="flex-shrink-0 text-slate-500 text-xs">{date}</div>
      </div>
      <StarRating value={review.rating} readOnly size="sm" />
      <p className="text-slate-400 text-sm leading-relaxed mt-3">{review.text}</p>
    </div>
  );
}

export default function ReviewsSection() {
  const [reviews, setReviews] = useState<Review[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  const [form, setForm] = useState({
    name: '',
    role: '',
    rating: 0,
    text: '',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  const fetchReviews = useCallback(async () => {
    const { data, error } = await supabase
      .from('reviews')
      .select('*')
      .order('created_at', { ascending: false });

    if (!error && data) setReviews(data as Review[]);
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchReviews();
  }, [fetchReviews]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(t);
  }, [toast]);

  const validate = () => {
    const errs: Record<string, string> = {};
    if (!form.name.trim()) errs.name = 'Name is required';
    if (form.rating === 0) errs.rating = 'Please select a rating';
    if (!form.text.trim()) errs.text = 'Review text is required';
    return errs;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length > 0) {
      setErrors(errs);
      return;
    }
    setErrors({});
    setSubmitting(true);

    const { error } = await supabase.from('reviews').insert([
      {
        name: form.name.trim(),
        role: form.role.trim(),
        rating: form.rating,
        text: form.text.trim(),
      },
    ]);

    setSubmitting(false);

    if (error) {
      setToast({ type: 'error', message: 'Failed to submit. Please try again.' });
    } else {
      setForm({ name: '', role: '', rating: 0, text: '' });
      setToast({ type: 'success', message: 'Thank you for your review!' });
      fetchReviews();
    }
  };

  const avgRating =
    reviews.length > 0
      ? reviews.reduce((acc, r) => acc + r.rating, 0) / reviews.length
      : 0;

  return (
    <section id="reviews" className="py-24 relative" aria-labelledby="reviews-title">
      <div
        className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-24 bg-gradient-to-b from-transparent to-cyan-400/30"
        aria-hidden="true"
      />

      {/* Toast */}
      {toast && (
        <div
          role="alert"
          aria-live="polite"
          className={`fixed bottom-6 right-6 z-50 px-5 py-3 rounded-xl text-sm font-medium shadow-lg transition-all duration-300 ${
            toast.type === 'success'
              ? 'bg-emerald-500/20 border border-emerald-500/40 text-emerald-300'
              : 'bg-red-500/20 border border-red-500/40 text-red-300'
          }`}
        >
          {toast.message}
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 mb-4 rounded-full border border-cyan-400/20 bg-cyan-400/5 text-cyan-400 text-xs font-medium uppercase tracking-wide">
            Community
          </div>
          <h2 id="reviews-title" className="text-3xl sm:text-4xl font-bold text-white mb-3">
            Reviews &amp; Feedback
          </h2>
          <p className="text-slate-400">
            Share your thoughts on PlantMirror with the team.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
          {/* Left: form */}
          <div>
            <div className="card-base rounded-2xl p-6 border border-[#1e2d45]">
              <h3 className="text-white font-semibold text-lg mb-6">Leave a Review</h3>
              <form onSubmit={handleSubmit} noValidate className="space-y-5">
                {/* Name */}
                <div>
                  <label htmlFor="review-name" className="block text-sm font-medium text-slate-300 mb-1.5">
                    Name <span className="text-red-400">*</span>
                  </label>
                  <input
                    id="review-name"
                    type="text"
                    value={form.name}
                    onChange={(e) => setForm({ ...form, name: e.target.value })}
                    placeholder="Your name"
                    className={`w-full px-4 py-2.5 rounded-xl bg-white/5 border text-slate-100 placeholder-slate-600 text-sm focus:outline-none focus:border-cyan-400/60 transition-colors ${
                      errors.name ? 'border-red-500/60' : 'border-[#1e2d45]'
                    }`}
                  />
                  {errors.name && <p className="text-red-400 text-xs mt-1">{errors.name}</p>}
                </div>

                {/* Role */}
                <div>
                  <label htmlFor="review-role" className="block text-sm font-medium text-slate-300 mb-1.5">
                    Affiliation / Role <span className="text-slate-600 text-xs">(optional)</span>
                  </label>
                  <input
                    id="review-role"
                    type="text"
                    value={form.role}
                    onChange={(e) => setForm({ ...form, role: e.target.value })}
                    placeholder="e.g. Reviewer, Professor, Engineer…"
                    className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-[#1e2d45] text-slate-100 placeholder-slate-600 text-sm focus:outline-none focus:border-cyan-400/60 transition-colors"
                  />
                </div>

                {/* Star rating */}
                <div>
                  <span className="block text-sm font-medium text-slate-300 mb-2">
                    Rating <span className="text-red-400">*</span>
                  </span>
                  <StarRating
                    value={form.rating}
                    onChange={(v) => setForm({ ...form, rating: v })}
                  />
                  {errors.rating && <p className="text-red-400 text-xs mt-1">{errors.rating}</p>}
                </div>

                {/* Review text */}
                <div>
                  <label htmlFor="review-text" className="block text-sm font-medium text-slate-300 mb-1.5">
                    Review <span className="text-red-400">*</span>
                  </label>
                  <textarea
                    id="review-text"
                    rows={4}
                    value={form.text}
                    onChange={(e) => setForm({ ...form, text: e.target.value })}
                    placeholder="Share your thoughts on PlantMirror…"
                    className={`w-full px-4 py-2.5 rounded-xl bg-white/5 border text-slate-100 placeholder-slate-600 text-sm focus:outline-none focus:border-cyan-400/60 transition-colors resize-none ${
                      errors.text ? 'border-red-500/60' : 'border-[#1e2d45]'
                    }`}
                  />
                  {errors.text && <p className="text-red-400 text-xs mt-1">{errors.text}</p>}
                </div>

                <button
                  type="submit"
                  disabled={submitting}
                  className="w-full py-3 rounded-xl bg-cyan-400 text-[#0b1220] font-semibold text-sm hover:bg-cyan-300 transition-all duration-200 glow-cyan disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {submitting ? 'Submitting…' : 'Submit Review'}
                </button>
              </form>
            </div>
          </div>

          {/* Right: reviews list */}
          <div>
            {/* Aggregate rating */}
            {reviews.length > 0 && (
              <div className="card-base rounded-2xl p-5 border border-[#1e2d45] mb-5 flex items-center gap-4">
                <div className="text-4xl font-bold gradient-text-cyan">{avgRating.toFixed(1)}</div>
                <div>
                  <StarRating value={Math.round(avgRating)} readOnly />
                  <p className="text-slate-500 text-xs mt-1">
                    Based on {reviews.length} review{reviews.length !== 1 ? 's' : ''}
                  </p>
                </div>
              </div>
            )}

            {/* Reviews list */}
            <div className="space-y-4 max-h-[600px] overflow-y-auto pr-1">
              {loading ? (
                <div className="space-y-4">
                  {[1, 2].map((i) => (
                    <div key={i} className="card-base rounded-xl p-5 border border-[#1e2d45] animate-pulse">
                      <div className="h-3 bg-white/5 rounded w-1/3 mb-3" />
                      <div className="h-3 bg-white/5 rounded w-1/4 mb-4" />
                      <div className="space-y-2">
                        <div className="h-2 bg-white/5 rounded" />
                        <div className="h-2 bg-white/5 rounded w-4/5" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : reviews.length === 0 ? (
                <div className="card-base rounded-2xl p-8 border border-[#1e2d45] text-center">
                  <div className="text-3xl mb-3" aria-hidden="true">💬</div>
                  <h4 className="text-white font-medium mb-2">No reviews yet</h4>
                  <p className="text-slate-500 text-sm">Be the first to share your thoughts on PlantMirror.</p>
                </div>
              ) : (
                reviews.map((review) => (
                  <ReviewCard key={review.id} review={review} />
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
