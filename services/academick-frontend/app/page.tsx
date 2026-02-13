// app/page.tsx
'use client';

import StudentHelper from '@/components/StudentHelper';

export default function Home() {
  return (
    <main className="min-h-screen p-4">
      <StudentHelper />
    </main>
  );
}

// components/StudentHelper.tsx
// Copy the StudentHelper component we created earlier here