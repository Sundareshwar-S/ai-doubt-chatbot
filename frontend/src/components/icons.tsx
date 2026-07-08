type IconProps = { className?: string };

export function CloudUploadIcon({ className }: IconProps) {
  return (
    <svg
      className={className}
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M7 18a4.5 4.5 0 0 1-.5-8.98 5.5 5.5 0 0 1 10.75-1.5A4 4 0 0 1 17 18H7Z" />
      <path d="M12 10v7" />
      <path d="m9 13 3-3 3 3" />
    </svg>
  );
}

export function BotIcon({ className }: IconProps) {
  return (
    <svg
      className={className}
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="4" y="9" width="16" height="11" rx="3" />
      <path d="M12 9V5" />
      <circle cx="12" cy="3.5" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="9" cy="14.5" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="15" cy="14.5" r="1.2" fill="currentColor" stroke="none" />
      <path d="M2 13v3" />
      <path d="M22 13v3" />
    </svg>
  );
}

export function UserIcon({ className }: IconProps) {
  return (
    <svg
      className={className}
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="8" r="4" />
      <path d="M4 20c0-3.87 3.58-7 8-7s8 3.13 8 7" />
    </svg>
  );
}
