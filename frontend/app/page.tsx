import { Dashboard, type Tab } from "@/components/Dashboard";

// Reads the query string on the server so the client starts in the right
// state: ?job=<id> reopens an upload's job (e.g. after a reload), ?tab=sample
// opens the sample clip.
export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  const params = await searchParams;
  const job = typeof params.job === "string" && /^[0-9a-f]{32}$/.test(params.job) ? params.job : null;
  const tab: Tab = params.tab === "sample" && !job ? "sample" : "upload";
  return <Dashboard initialTab={tab} initialJobId={job} />;
}
