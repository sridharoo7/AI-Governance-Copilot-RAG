import "./styles.css";

/** Provides the shared shell for the evidence-grounded chat experience. */
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}

