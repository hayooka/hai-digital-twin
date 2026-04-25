'use client';

import Image from 'next/image';

interface TeamMember {
  name: string;
  role: string;
  tag?: string;
  initials: string;
}

const supervisor: TeamMember = {
  name: 'Professor Khaled Chahine',
  role: 'Supervisor',
  tag: 'Electrical Engineering Department',
  initials: 'KC',
};

const students: TeamMember[] = [
  { name: 'Farah AlHayek', role: 'Computer Engineer', initials: 'FA' },
  { name: 'Mudhi Aldihani', role: 'Electrical Engineer', initials: 'MA' },
  { name: 'Bedour Mahdi', role: 'Computer Engineer', initials: 'BM' },
  { name: 'Fatma Abdulaziz', role: 'Electrical Engineer', initials: 'FA' },
  { name: 'Mariam Alsahhaf', role: 'Computer Engineer', initials: 'MS' },
];

const avatarColors = [
  'from-cyan-500 to-teal-600',
  'from-blue-500 to-cyan-600',
  'from-teal-500 to-emerald-600',
  'from-amber-500 to-orange-600',
  'from-sky-500 to-blue-600',
];


function MemberCard({
  member,
  colorClass,
  isLarge = false,
}: {
  member: TeamMember;
  colorClass?: string;
  isLarge?: boolean;
}) {
  return (
    <div
      className={`card-base card-hover rounded-2xl flex flex-col items-center text-center ${
        isLarge ? 'p-8' : 'p-6'
      } border border-[#1e2d45] section-reveal`}
    >
      {/* Avatar */}
      <div
        className={`relative flex-shrink-0 mb-4 ${isLarge ? 'w-20 h-20' : 'w-16 h-16'} rounded-full bg-gradient-to-br ${colorClass || 'from-cyan-500 to-teal-600'} flex items-center justify-center text-white font-bold ${isLarge ? 'text-xl' : 'text-base'} ${isLarge ? 'ring-2 ring-cyan-400/40 glow-cyan' : ''}`}
        aria-hidden="true"
      >
        {member.initials}
      </div>

      <h3 className={`font-semibold text-white mb-1 ${isLarge ? 'text-xl' : 'text-base'}`}>
        {member.name}
      </h3>

      <span
        className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-medium mb-2 ${
          isLarge
            ? 'bg-cyan-400/15 text-cyan-400 border border-cyan-400/30'
            : 'bg-slate-700/60 text-slate-300'
        }`}
      >
        {member.role}
      </span>

      {member.tag && (
        <p className="text-slate-500 text-xs mb-4">{member.tag}</p>
      )}

    </div>
  );
}

export default function TeamSection() {
  return (
    <section id="team" className="py-24 relative" aria-labelledby="team-title">
      <div
        className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-24 bg-gradient-to-b from-transparent to-cyan-400/30"
        aria-hidden="true"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section header */}
        <div className="text-center mb-16">
<h2 id="team-title" className="text-3xl sm:text-4xl font-bold text-white mb-3">
            The Team
          </h2>

        </div>

        {/* Supervisor — spotlit, centred */}
        <div className="flex justify-center mb-12">
          <div className="w-full max-w-sm">
            <div className="mb-3 text-center">
              <span className="text-xs font-semibold text-amber-400 uppercase tracking-widest">
                Project Supervisor
              </span>
            </div>
            <MemberCard
              member={supervisor}
              colorClass="from-amber-500 to-orange-600"
              isLarge
            />
          </div>
        </div>

        {/* Student divider label */}
        <div className="flex items-center gap-4 mb-8">
          <div className="flex-1 h-px bg-[#1e2d45]" />
          <span className="text-slate-500 text-xs font-medium uppercase tracking-widest">Students</span>
          <div className="flex-1 h-px bg-[#1e2d45]" />
        </div>

        {/* Students grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-5">
          {students.map((student, i) => (
            <MemberCard
              key={student.name}
              member={student}
              colorClass={avatarColors[i]}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
