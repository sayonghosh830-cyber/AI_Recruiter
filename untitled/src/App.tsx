import { useState, useEffect, ChangeEvent } from "react";
import { 
  Sparkles, 
  Users, 
  Award, 
  ShieldAlert, 
  Upload, 
  Download, 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  Search, 
  Settings, 
  ChevronDown, 
  ChevronUp, 
  FileText, 
  Cpu, 
  FileJson,
  TrendingUp,
  Briefcase
} from "lucide-react";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  Cell, 
  CartesianGrid 
} from "recharts";

interface CandidateRecord {
  candidate_id: string;
  name: string;
  leaderboard_rank: string | number;
  final_match_score: string | number;
  hiring_confidence: string;
  one_liner_reasoning: string;
  suggested_action: string;
  matched_skills: string;
  missing_skills: string;
  stability_score?: string | number;
  academic_prestige?: string | number;
  platform_responsiveness?: string | number;
  attendance_reliability?: string | number;
  churn_risk?: string | number;
  notice_period_days?: string | number;
}

interface RawCandidateProfile {
  candidate_id: string;
  profile: {
    anonymized_name: string;
    headline: string;
    summary: string;
    location: string;
    country: string;
    years_of_experience: number;
    current_title: string;
    current_company: string;
    current_company_size: string;
    current_industry: string;
  };
  skills: Array<{ name: string; proficiency: string; endorsements: number; duration_months: number }>;
  redrob_signals?: {
    notice_period_days?: number;
    offer_acceptance_rate?: number;
  };
}

export default function App() {
  // State variables
  const [candidates, setCandidates] = useState<RawCandidateProfile[]>([]);
  const [rankedRecords, setRankedRecords] = useState<CandidateRecord[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [successText, setSuccessText] = useState<string | null>(null);
  const [searchText, setSearchText] = useState<string>("");
  const [expandedCardId, setExpandedCardId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"leaderboard" | "raw_catalog">("leaderboard");
  
  // Job Reference from config.json (statically loaded for convenience/speed)
  const targetJob = {
    title: "Backend Software Engineer",
    description: "We are seeking a highly experienced Backend Developer expert in Python, SQL databases and microservice design. Experience building distributed queries, streaming pipelines with Apache Spark/Kafka and handling real-time data integration is a major plus.",
    skills: ["Python", "SQL", "Kafka", "Spark", "Docker"]
  };

  // 1. Initial Load: Fetch candidate profiles catalog of dataset
  useEffect(() => {
    fetchCandidates();
  }, []);

  const fetchCandidates = async () => {
    try {
      const res = await fetch("/api/candidates");
      if (!res.ok) {
        throw new Error(`Failed to load base candidate catalog. Status: ${res.status}`);
      }
      const data = await res.json();
      setCandidates(data);
    } catch (err: any) {
      console.error(err);
      setErrorText(`Error fetching candidates catalog: ${err.message}`);
    }
  };

  // 2. Trigger pipeline ranking
  const runRankingPipeline = async (uploadPayload?: RawCandidateProfile[]) => {
    setIsLoading(true);
    setErrorText(null);
    setSuccessText(null);
    try {
      const res = await fetch("/api/rank", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          candidates: uploadPayload || null // If null, the backend uses existing candidates catalog
        }),
      });

      const body = await res.json();
      if (!res.ok) {
        throw new Error(body.error || body.details || "Server failed while ranking candidates.");
      }

      const recordsWithNumbers = body.records.map((r: any) => ({
        ...r,
        leaderboard_rank: Number(r.leaderboard_rank),
        final_match_score: Number(r.final_match_score),
      }));

      setRankedRecords(recordsWithNumbers);
      setSuccessText("Successfully parsed and ranked candidates!");
      setActiveTab("leaderboard");
      
      // If we ranked custom uploaded files, re-fetch the raw state as well to sync UI views
      if (uploadPayload) {
        setCandidates(uploadPayload);
      } else {
        fetchCandidates();
      }
    } catch (err: any) {
      console.error(err);
      setErrorText(err.message || "An unexpected error occurred during ranking.");
    } finally {
      setIsLoading(false);
    }
  };

  // 3. Handle Drag & Drop / File selection logic
  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const parsed = JSON.parse(text);
        
        if (!Array.isArray(parsed)) {
          throw new Error("Invalid candidates schema: Must be a JSON array of profile objects.");
        }
        
        // Match schema basics
        if (parsed.length > 0 && (!parsed[0].profile || !parsed[0].candidate_id)) {
          throw new Error("Invalid format. Missing 'candidate_id' or 'profile' blocks inside elements.");
        }

        setErrorText(null);
        setSuccessText(`Loaded custom file: ${file.name} (${parsed.length} items parsed). Click 'Rank' to run weights!`);
        setCandidates(parsed);
      } catch (err: any) {
        setErrorText(`File upload parser error: ${err.message}`);
      }
    };
    reader.readAsText(file);
  };

  // 4. Download processed CSV output directly
  const triggerCSVDownload = () => {
    if (rankedRecords.length === 0) return;
    
    // Assemble simple CSV string
    const headers = [
      "candidate_id", "name", "leaderboard_rank", "final_match_score", 
      "hiring_confidence", "one_liner_reasoning", "suggested_action", 
      "matched_skills", "missing_skills", "stability_score", "academic_prestige", "platform_responsiveness"
    ];

    const lines = [headers.join(",")];
    
    rankedRecords.forEach((row) => {
      const lineValues = headers.map((header) => {
        let val = (row as any)[header] ?? "";
        // Clean double quotes
        val = String(val).replace(/"/g, '""');
        if (val.includes(",") || val.includes("\n") || val.includes('"')) {
          return `"${val}"`;
        }
        return val;
      });
      lines.push(lineValues.join(","));
    });

    const csvContent = "data:text/csv;charset=utf-8," + encodeURIComponent(lines.join("\n"));
    const link = document.createElement("a");
    link.getAnimations();
    link.href = csvContent;
    link.download = "ranked_leaderboard.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Calculation helpers for KPI Cards
  const totalProfiles = candidates.length;
  
  const topScore = rankedRecords.length > 0 
    ? Math.max(...rankedRecords.map(r => Number(r.final_match_score))) 
    : 0;

  const averageScore = rankedRecords.length > 0 
    ? rankedRecords.reduce((acc, curr) => acc + Number(curr.final_match_score), 0) / rankedRecords.length 
    : 0;

  const recommendedCount = rankedRecords.filter(r => 
    r.hiring_confidence === "EXCEPTIONAL" || r.hiring_confidence === "RECOMMENDED"
  ).length;

  // Preparing bar charts metrics
  const sortedGraphData = [...rankedRecords]
    .sort((a, b) => Number(b.final_match_score) - Number(a.final_match_score))
    .slice(0, 10)
    .map(r => ({
      name: r.name && r.name !== "Anonymized Candidate" ? r.name.split(" ")[0] : r.candidate_id,
      score: Number(r.final_match_score),
      confidence: r.hiring_confidence
    }));

  const tierCountsMap = rankedRecords.reduce((acc: any, curr) => {
    acc[curr.hiring_confidence] = (acc[curr.hiring_confidence] || 0) + 1;
    return acc;
  }, {});

  const distributionData = [
    { name: "Exceptional", count: tierCountsMap["EXCEPTIONAL"] || 0, fill: "#3fb950" },
    { name: "Recommended", count: tierCountsMap["RECOMMENDED"] || 0, fill: "#58a6ff" },
    { name: "Caution", count: tierCountsMap["CONSIDER_WITH_CAUTION"] || 0, fill: "#d29922" },
    { name: "Not Aligned", count: tierCountsMap["NOT_ALIGNED"] || 0, fill: "#f85149" }
  ];

  // Filters candidates list for search query matches
  const filteredRankedResult = rankedRecords.filter(r => 
    r.name.toLowerCase().includes(searchText.toLowerCase()) ||
    r.candidate_id.toLowerCase().includes(searchText.toLowerCase()) ||
    r.hiring_confidence.toLowerCase().includes(searchText.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-[#090d13] text-[#c9d1d9] flex flex-col font-sans transition-colors duration-200">
      
      {/* 1. Header Layout */}
      <header className="border-b border-[#21262d] bg-[#161b22] px-6 py-4 flex items-center justify-between sticky top-0 z-50 shadow-md">
        <div className="flex items-center space-x-3">
          <div className="bg-[#1f2937] p-2 rounded-lg border border-[#30363d] shadow-indigo-500/20 shadow-inner">
            <Briefcase className="w-6 h-6 text-[#58a6ff]" />
          </div>
          <div>
            <h1 className="text-xl font-bold font-display tracking-wide text-white flex items-center gap-2">
              Recruiter Hybrid Ranking Dashboard
              <span className="text-xs font-mono font-normal bg-brand-bg/80 border border-[#30363d] px-2 py-0.5 rounded text-indigo-300">v1.2.0</span>
            </h1>
            <p className="text-xs text-[#8b949e]">End-to-End semantic calibration and multi-criteria priority scoring</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-xs text-[#8b949e] bg-[#0d1117] border border-[#21262d] px-3 py-1.5 rounded-full shadow-inner">
            <span className="w-2.5 h-2.5 bg-[#3fb950] rounded-full animate-pulse"></span>
            <span className="font-mono text-white">Pipeline Sync: Connected</span>
          </div>
        </div>
      </header>

      {/* Main Grid Content Layout */}
      <main className="flex-1 w-full max-w-7xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Sidebar Interactive Tools */}
        <div className="lg:col-span-4 flex flex-col space-y-6">
          
          {/* Target Job Config Parameters Panel */}
          <section className="bg-[#151b23] border border-[#30363d] rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold tracking-wider uppercase text-slate-400 flex items-center gap-2">
                <Settings className="w-4 h-4 text-[#58a6ff]" /> Target Job Parameters
              </h2>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="text-[10px] font-mono text-[#8b949e] uppercase font-bold">Target Position</label>
                <div className="text-white font-medium text-sm mt-0.5 bg-[#0d1117] px-3 py-2 rounded border border-[#21262d]">
                  {targetJob.title}
                </div>
              </div>

              <div>
                <label className="text-[10px] font-mono text-[#8b949e] uppercase font-bold">Required Stacks / Core Skills</label>
                <div className="flex flex-wrap gap-1.5 mt-1.5">
                  {targetJob.skills.map((skill) => (
                    <span 
                      key={skill} 
                      className="text-xs font-mono px-2 py-1 bg-[#1f2937] hover:bg-[#30363d] text-[#58a6ff] rounded border border-indigo-900/60 transition-all duration-150"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>

              <div className="bg-[#0d1117] border border-[#21262d] rounded p-3 text-xs text-[#8b949e] leading-relaxed">
                <span className="font-semibold text-white block mb-1">Semantic Rule Context:</span>
                {targetJob.description}
              </div>
            </div>
          </section>

          {/* Interactive File Uploader and Catalog Actions */}
          <section className="bg-[#151b23] border border-[#30363d] rounded-xl p-5 shadow-sm">
            <h2 className="text-sm font-semibold tracking-wider uppercase text-slate-400 mb-4 flex items-center gap-2">
              <Upload className="w-4 h-4 text-[#58a6ff]" /> Candidate Data Ingestion
            </h2>

            <div className="space-y-4">
              {/* File uploading area container */}
              <div className="border-2 border-dashed border-[#30363d] hover:border-[#58a6ff] rounded-xl p-6 text-center cursor-pointer transition-all bg-[#0d1117]/50 relative group">
                <input 
                  type="file" 
                  accept=".json"
                  onChange={handleFileUpload} 
                  className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-10" 
                />
                <div className="flex flex-col items-center justify-center space-y-2">
                  <div className="bg-[#161b22] p-2.5 rounded-lg border border-[#30363d] group-hover:border-[#58a6ff] transition-colors">
                    <FileJson className="w-6 h-6 text-[#8b949e] group-hover:text-[#58a6ff]" />
                  </div>
                  <div>
                    <span className="text-xs text-white font-medium block">Upload Candidate Profiles (.json)</span>
                    <span className="text-[10px] text-[#8b949e] block mt-1">Accepts schemas compatible with system schema</span>
                  </div>
                </div>
              </div>

              {/* Status messages info */}
              {errorText && (
                <div className="bg-red-950/30 border border-red-900/50 rounded-lg p-3 text-xs text-red-400 flex items-start space-x-2">
                  <XCircle className="w-4 h-4 mt-0.5 shrink-0 text-[#f85149]" />
                  <span>{errorText}</span>
                </div>
              )}

              {successText && (
                <div className="bg-green-950/20 border border-green-900/40 rounded-lg p-3 text-xs text-green-400 flex items-start space-x-2">
                  <CheckCircle className="w-4 h-4 mt-0.5 shrink-0 text-[#3fb950]" />
                  <span>{successText}</span>
                </div>
              )}

              {/* Primary ranking triggers */}
              <button
                onClick={() => runRankingPipeline()}
                disabled={isLoading}
                className="w-full bg-[#238636] hover:bg-[#2ea043] disabled:bg-[#238636]/40 text-white font-medium py-3 px-4 rounded-lg shadow transition-all duration-150 flex items-center justify-center space-x-2 cursor-pointer disabled:cursor-not-allowed hover:shadow-green-900/20"
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span className="font-display">Computing Match Metrics...</span>
                  </>
                ) : (
                  <>
                    <Cpu className="w-4 h-4" />
                    <span className="font-display">Rank Candidates</span>
                  </>
                )}
              </button>
            </div>
          </section>

        </div>

        {/* Right Dashboard Results View */}
        <div className="lg:col-span-8 space-y-6">
          
          {/* 3. KPI Statistics Cards Rows */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            
            <div className="bg-[#151b23] border border-[#30363d] rounded-xl p-4 flex flex-col justify-between shadow-sm hover:border-slate-600 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold font-mono tracking-wider text-[#8b949e] uppercase">Total Profiles</span>
                <Users className="w-4 h-4 text-[#8b949e]" />
              </div>
              <div>
                <div className="text-2xl font-bold font-display text-white">{totalProfiles}</div>
                <p className="text-[10px] text-[#8b949e] mt-1">Loaded profiles</p>
              </div>
            </div>

            <div className="bg-[#151b23] border border-[#30363d] rounded-xl p-4 flex flex-col justify-between shadow-sm hover:border-slate-600 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold font-mono tracking-wider text-[#8b949e] uppercase">Top Match Score</span>
                <Award className="w-4 h-4 text-[#d29922]" />
              </div>
              <div>
                <div className="text-2xl font-bold font-display text-[#d29922]">
                  {topScore > 0 ? topScore.toFixed(4) : "0.00"}
                </div>
                <p className="text-[10px] text-[#8b949e] mt-1">Highest priority</p>
              </div>
            </div>

            <div className="bg-[#151b23] border border-[#30363d] rounded-xl p-4 flex flex-col justify-between shadow-sm hover:border-slate-600 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold font-mono tracking-wider text-[#8b949e] uppercase">Average Match</span>
                <TrendingUp className="w-4 h-4 text-[#58a6ff]" />
              </div>
              <div>
                <div className="text-2xl font-bold font-display text-[#58a6ff]">
                  {averageScore > 0 ? averageScore.toFixed(4) : "0.00"}
                </div>
                <p className="text-[10px] text-[#8b949e] mt-1">Cohort baseline</p>
              </div>
            </div>

            <div className="bg-[#151b23] border border-[#30363d] rounded-xl p-4 flex flex-col justify-between shadow-sm hover:border-[#3fb950] transition-colors">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold font-mono tracking-wider text-[#8b949e] uppercase font-display">Recommended</span>
                <Sparkles className="w-4 h-4 text-[#3fb950]" />
              </div>
              <div>
                <div className="text-2xl font-bold font-display text-[#3fb950]">{recommendedCount}</div>
                <p className="text-[10px] text-[#8b949e] mt-1">Critical high fit</p>
              </div>
            </div>

          </div>

          {/* 4. Score Distribution Charts (Shown only when records exist) */}
          {rankedRecords.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-[#151b23] border border-[#30363d] rounded-xl p-5 shadow-sm">
              
              <div>
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Top 10 Rankings</h3>
                <div className="h-44 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={sortedGraphData} margin={{ top: 5, right: 5, left: -25, bottom: 5 }}>
                      <XAxis dataKey="name" stroke="#8b949e" fontSize={10} tickLine={false} />
                      <YAxis stroke="#8b949e" fontSize={10} domain={[0, 1]} tickLine={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d', borderRadius: 6 }} 
                        labelStyle={{ color: 'white', fontWeight: 600 }}
                      />
                      <Bar dataKey="score">
                        {sortedGraphData.map((entry, index) => {
                          let color = "#58a6ff"; // Recommended - Blue
                          if (entry.confidence === "EXCEPTIONAL") color = "#3fb950"; // Exceptional - Green
                          else if (entry.confidence === "CONSIDER_WITH_CAUTION") color = "#d29922"; // Caution - Gold
                          else if (entry.confidence === "NOT_ALIGNED") color = "#f85149"; // Not Aligned - Red
                          return <Cell key={`cell-${index}`} fill={color} opacity={0.8} />;
                        })}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div>
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Recommendation Tiers Distribution</h3>
                <div className="h-44 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={distributionData} layout="vertical" margin={{ top: 5, right: 5, left: -10, bottom: 5 }}>
                      <XAxis type="number" stroke="#8b949e" fontSize={10} hide />
                      <YAxis type="category" dataKey="name" stroke="#8b949e" fontSize={10} tickLine={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d', borderRadius: 6 }}
                        itemStyle={{ color: '#fff' }}
                      />
                      <Bar dataKey="count">
                        {distributionData.map((entry, index) => (
                          <Cell key={`cell-dist-${index}`} fill={entry.fill} opacity={0.85} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

            </div>
          )}

          {/* Catalog & Leaderboard Toggle tabs section */}
          <div>
            <div className="flex items-center justify-between border-b border-[#21262d] mb-4 pb-2">
              <div className="flex space-x-4">
                <button
                  onClick={() => setActiveTab("leaderboard")}
                  className={`text-sm font-medium pb-2 border-b-2 transition-all cursor-pointer ${
                    activeTab === "leaderboard" 
                      ? "border-[#58a6ff] text-white" 
                      : "border-transparent text-[#8b949e] hover:text-slate-300"
                  }`}
                >
                  🏆 Priority Leaderboard {rankedRecords.length > 0 && `(${rankedRecords.length})`}
                </button>
                <button
                  onClick={() => setActiveTab("raw_catalog")}
                  className={`text-sm font-medium pb-2 border-b-2 transition-all cursor-pointer ${
                    activeTab === "raw_catalog" 
                      ? "border-[#58a6ff] text-white" 
                      : "border-transparent text-[#8b949e] hover:text-slate-300"
                  }`}
                >
                  📁 Candidate Ingested List ({candidates.length})
                </button>
              </div>

              {rankedRecords.length > 0 && activeTab === "leaderboard" && (
                <button
                  onClick={triggerCSVDownload}
                  className="bg-[#21262d] hover:bg-[#30363d] text-[#c9d1d9] border border-[#30363d] px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-colors flex items-center space-x-1.5 active:bg-[#161b22]"
                >
                  <Download className="w-3.5 h-3.5 text-[#58a6ff]" />
                  <span>Download Structured CSV</span>
                </button>
              )}
            </div>

            {/* TAB CONTENT: LEADERBOARD RESULTS */}
            {activeTab === "leaderboard" && (
              <div className="space-y-4">
                {rankedRecords.length === 0 ? (
                  <div className="bg-[#151b23] border border-dashed border-[#30363d] rounded-xl p-12 text-center">
                    <Cpu className="w-10 h-10 text-slate-500 mx-auto mb-3 animate-pulse" />
                    <h3 className="text-white text-base font-semibold mb-1">Prioritization Weights Stale</h3>
                    <p className="text-xs text-[#8b949e] max-w-md mx-auto leading-relaxed mb-4">
                      The candidate prioritisation matrix hasn't been executed for this cohort. Add candidates or click the Rank Candidates button to start.
                    </p>
                    <button
                      onClick={() => runRankingPipeline()}
                      disabled={isLoading}
                      className="bg-[#1f2937] hover:bg-[#30363d] text-white border border-[#30363d] text-xs font-medium py-2 px-6 rounded-lg cursor-pointer transition-colors"
                    >
                      Initialize Default Cohort Ranking
                    </button>
                  </div>
                ) : (
                  <>
                    {/* Search filter banner */}
                    <div className="relative">
                      <Search className="w-4 h-4 text-[#8b949e] absolute left-3.5 top-1/2 -translate-y-1/2" />
                      <input
                        type="text"
                        placeholder="Search leaderboard by candidate name, ID, or confidence tier..."
                        value={searchText}
                        onChange={(e) => setSearchText(e.target.value)}
                        className="w-full bg-[#0d1117] border border-[#30363d] hover:border-[#58a6ff] focus:border-[#58a6ff] focus:outline-none rounded-xl pl-10 pr-4 py-2.5 text-xs text-white placeholder-slate-500 transition-colors"
                      />
                    </div>

                    {/* Top 25 Cand Table summary layout */}
                    <div className="bg-[#151b23] border border-[#30363d] rounded-xl overflow-hidden shadow-sm">
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs border-collapse">
                          <thead>
                            <tr className="bg-[#161b22] border-b border-[#30363d] text-[#8b949e] uppercase text-[10px] tracking-wider font-semibold font-mono">
                              <th className="px-4 py-3 text-center">Rank</th>
                              <th className="px-4 py-3">Profile ID</th>
                              <th className="px-4 py-3">FullName</th>
                              <th className="px-4 py-3 text-center">Priority</th>
                              <th className="px-4 py-3">Hiring Rec</th>
                              <th className="px-4 py-3 text-right">Details</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-[#21262d]">
                            {filteredRankedResult.slice(0, 25).map((row, index) => {
                              // Define row badges style
                              let badgeColor = "bg-red-950/40 text-red-400 border border-red-900/40";
                              if (row.hiring_confidence === "EXCEPTIONAL") badgeColor = "bg-green-950/40 text-green-400 border border-green-900/40";
                              else if (row.hiring_confidence === "RECOMMENDED") badgeColor = "bg-blue-950/40 text-blue-400 border border-blue-900/40";
                              else if (row.hiring_confidence === "CONSIDER_WITH_CAUTION") badgeColor = "bg-amber-950/40 text-amber-500 border border-amber-900/40";

                              return (
                                <tr key={row.candidate_id} className="hover:bg-[#1c2128] transition-colors duration-150">
                                  <td className="px-4 py-3.5 text-center font-bold text-white font-mono">{row.leaderboard_rank}</td>
                                  <td className="px-4 py-3.5 font-mono text-[#58a6ff]">{row.candidate_id}</td>
                                  <td className="px-4 py-3.5 font-semibold text-white">{row.name}</td>
                                  <td className="px-4 py-3.5 text-center font-mono font-bold text-indigo-400">
                                    {Number(row.final_match_score).toFixed(4)}
                                  </td>
                                  <td className="px-4 py-3.5">
                                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${badgeColor}`}>
                                      {row.hiring_confidence}
                                    </span>
                                  </td>
                                  <td className="px-4 py-3.5 text-right">
                                    <button
                                      onClick={() => setExpandedCardId(expandedCardId === row.candidate_id ? null : row.candidate_id)}
                                      className="text-slate-400 hover:text-white px-2 py-1 bg-[#161b22] border border-[#30363d] rounded cursor-pointer text-[10px] uppercase font-bold tracking-wider hover:border-[#58a6ff] transition-all"
                                    >
                                      {expandedCardId === row.candidate_id ? "Collapse" : "Expand"}
                                    </button>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Expandable candidate profile detailed listing cards */}
                    <div className="space-y-4 pt-2">
                      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest block font-display">Expandable Evaluation Records</h3>
                      {filteredRankedResult.slice(0, 25).map((row, index) => {
                        const isExpanded = expandedCardId === row.candidate_id;

                        let badgeColorClass = "border-red-900/60 bg-red-950/20 text-red-500";
                        if (row.hiring_confidence === "EXCEPTIONAL") badgeColorClass = "border-[#2ea043]/60 bg-green-950/20 text-green-400";
                        else if (row.hiring_confidence === "RECOMMENDED") badgeColorClass = "border-[#58a6ff]/60 bg-blue-950/20 text-blue-400";
                        else if (row.hiring_confidence === "CONSIDER_WITH_CAUTION") badgeColorClass = "border-amber-900/60 bg-amber-950/20 text-yellow-400";

                        const matchedArr = row.matched_skills ? row.matched_skills.split(", ") : [];
                        const missingArr = row.missing_skills ? row.missing_skills.split(", ") : [];

                        return (
                          <div 
                            key={`card-${row.candidate_id}`} 
                            className={`bg-[#151b23] border ${isExpanded ? "border-[#58a6ff] shadow-lg" : "border-[#30363d]"} rounded-xl overflow-hidden transition-all duration-200 hover:border-slate-500`}
                          >
                            <div 
                              onClick={() => setExpandedCardId(isExpanded ? null : row.candidate_id)}
                              className="px-5 py-4 flex items-center justify-between cursor-pointer select-none"
                            >
                              <div className="flex items-center space-x-3.5">
                                <div className="text-base font-bold font-mono text-slate-400">#{row.leaderboard_rank}</div>
                                <div>
                                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                                    {row.name} 
                                    <span className="text-xs font-mono font-normal text-slate-400">({row.candidate_id})</span>
                                  </h4>
                                  <p className="text-[10px] text-[#8b949e] font-sans mt-0.5 max-w-md line-clamp-1">{row.one_liner_reasoning}</p>
                                </div>
                              </div>
                              <div className="flex items-center space-x-4">
                                <div className="text-right">
                                  <div className="text-sm font-bold text-[#5c9dff] font-mono">{Number(row.final_match_score).toFixed(4)}</div>
                                  <div className="text-[10px] text-slate-500 font-mono mt-0.5">Match Index</div>
                                </div>
                                <span className={`px-2.5 py-1 text-[10px] font-bold rounded border ${badgeColorClass}`}>
                                  {row.hiring_confidence}
                                </span>
                                {isExpanded ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
                              </div>
                            </div>

                            {/* Collapsible expanded section of detail dashboard diagnostics */}
                            {isExpanded && (
                              <div className="px-5 pb-5 pt-1 border-t border-[#21262d] bg-[#0c1017]/40 space-y-4">
                                
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                  
                                  {/* Matched skills lists */}
                                  <div className="bg-[#151b23] border border-[#21262d] p-3 rounded-lg">
                                    <h5 className="text-[10px] font-bold font-mono uppercase tracking-wider text-green-500 flex items-center gap-1 mb-2">
                                      <CheckCircle className="w-3.5 h-3.5" /> Checked Stacks & Match Alignments
                                    </h5>
                                    {matchedArr.length > 0 && matchedArr[0] !== "" ? (
                                      <div className="flex flex-wrap gap-1.5">
                                        {matchedArr.map(m => (
                                          <span key={m} className="text-[11px] font-mono bg-green-950/20 text-green-400 border border-green-900/40 px-2 py-0.5 rounded">
                                            {m}
                                          </span>
                                        ))}
                                      </div>
                                    ) : (
                                      <p className="text-xs text-[#8b949e]">No target stack matches detected in historical catalog.</p>
                                    )}
                                  </div>

                                  {/* Missing targets */}
                                  <div className="bg-[#151b23] border border-[#21262d] p-3 rounded-lg">
                                    <h5 className="text-[10px] font-bold font-mono uppercase tracking-wider text-[#d29922] flex items-center gap-1 mb-2">
                                      <AlertTriangle className="w-3.5 h-3.5" /> Identified Skill / Tool Gaps
                                    </h5>
                                    {missingArr.length > 0 && missingArr[0] !== "" ? (
                                      <div className="flex flex-wrap gap-1.5">
                                        {missingArr.map(m => (
                                          <span key={m} className="text-[11px] font-mono bg-amber-950/20 text-yellow-500 border border-amber-900/40 px-2 py-0.5 rounded">
                                            {m}
                                          </span>
                                        ))}
                                      </div>
                                    ) : (
                                      <p className="text-xs text-[#3fb950] font-medium">All target job skills successfully filled.</p>
                                    )}
                                  </div>

                                </div>

                                {/* AI Rationale details description blocks */}
                                <div className="space-y-2">
                                  <div className="text-[10px] font-mono uppercase font-bold tracking-wider text-slate-500">Evaluation Strategy & Reasoning</div>
                                  <blockquote className="border-l-2 border-indigo-500 bg-[#161c24] text-xs text-slate-200 px-4 py-3 rounded-r-lg italic leading-relaxed">
                                    "{row.one_liner_reasoning}"
                                  </blockquote>
                                </div>

                                <div className="space-y-2">
                                  <div className="text-[10px] font-mono uppercase font-bold tracking-wider text-slate-500">Suggested Recruiter Hiring Procedure</div>
                                  <div className="text-xs font-semibold text-white bg-[#0f141c] px-4 py-2 rounded-lg border border-[#30363d] flex items-center gap-2">
                                    <Sparkles className="w-3.5 h-3.5 text-[#3fb950] max-w-none shadow-md" />
                                    <span>{row.suggested_action}</span>
                                  </div>
                                </div>

                                {/* Custom Signals/Metrics Breakdown diagnostics */}
                                {row.stability_score !== undefined && (
                                  <div className="pt-2">
                                    <div className="text-[10px] font-mono uppercase font-bold tracking-wider text-slate-500 mb-2">Diagnostic Metrics Breakdown:</div>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                      <div className="p-2.5 rounded bg-[#10141b] border border-[#21262d] text-center">
                                        <span className="text-[10px] text-[#8b949e] uppercase font-mono tracking-wider block">Stability Score</span>
                                        <span className="text-xs font-bold text-white font-mono mt-0.5 block">{row.stability_score}</span>
                                      </div>
                                      <div className="p-2.5 rounded bg-[#10141b] border border-[#21262d] text-center">
                                        <span className="text-[10px] text-[#8b949e] uppercase font-mono tracking-wider block">Academic Merit</span>
                                        <span className="text-xs font-bold text-white font-mono mt-0.5 block">{row.academic_prestige}</span>
                                      </div>
                                      <div className="p-2.5 rounded bg-[#10141b] border border-[#21262d] text-center">
                                        <span className="text-[10px] text-[#8b949e] uppercase font-mono tracking-wider block">Response Rate</span>
                                        <span className="text-xs font-bold text-white font-mono mt-0.5 block">{(Number(row.platform_responsiveness) * 100).toFixed(0)}%</span>
                                      </div>
                                      <div className="p-2.5 rounded bg-[#10141b] border border-[#21262d] text-center">
                                        <span className="text-[10px] text-[#8b949e] uppercase font-mono tracking-wider block">Notice Period</span>
                                        <span className="text-xs font-bold text-white font-mono mt-0.5 block">{row.notice_period_days || "Immediate"} d</span>
                                      </div>
                                    </div>
                                  </div>
                                )}

                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            )}

            {/* TAB CONTENT: RAW CANDIDATES LIST CATALOG */}
            {activeTab === "raw_catalog" && (
              <div className="space-y-4">
                <div className="bg-[#151b23] border border-[#30363d] rounded-xl p-5 shadow-sm space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold text-white tracking-wide text-xs uppercase font-display">Loaded Raw Candidate Catalog</h3>
                      <p className="text-[10px] text-[#8b949e]">Relational profiles pending priorities weighting calibration</p>
                    </div>
                    <button 
                      onClick={() => runRankingPipeline(candidates)}
                      disabled={isLoading}
                      className="bg-[#58a6ff]/25 border border-[#58a6ff]/40 text-[#58a6ff] hover:bg-[#58a6ff]/40 px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-colors"
                    >
                      Rank This Ingested Catalog
                    </button>
                  </div>

                  <div className="divide-y divide-[#31363e]/60 space-y-3.5 max-h-[600px] overflow-y-auto pr-2">
                    {candidates.map((cand) => (
                      <div key={cand.candidate_id} className="pt-3.5 first:pt-0">
                        <div className="flex justify-between items-start">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-mono font-bold text-[#58a6ff]">{cand.candidate_id}</span>
                              <h4 className="text-xs font-bold text-white">{cand.profile.anonymized_name}</h4>
                            </div>
                            <span className="text-[11px] font-semibold text-slate-300 block font-sans mt-0.5">{cand.profile.headline}</span>
                            <p className="text-[10px] text-[#8b949e] tracking-tight leading-relaxed max-w-xl mt-1.5">{cand.profile.summary}</p>
                          </div>
                          <div className="text-right">
                            <span className="text-xs text-slate-400 font-mono font-medium block">{cand.profile.years_of_experience} yrs exp</span>
                            <span className="text-[10px] text-[#8b949e] font-mono mt-0.5 block">{cand.profile.location}</span>
                          </div>
                        </div>
                        <div className="flex items-center justify-between mt-3 bg-[#0d1117] p-2 rounded border border-[#21262d] text-[10px]">
                          <div className="flex flex-wrap gap-1">
                            {cand.skills.map((s, idx) => (
                              <span key={`${cand.candidate_id}-skill-${idx}`} className="font-mono bg-indigo-950/15 border border-[#21262d] text-slate-300 px-1.5 py-0.5 rounded">
                                {s.name} ({s.proficiency})
                              </span>
                            ))}
                          </div>
                          {cand.redrob_signals && (
                            <div className="text-[#8b949e] font-mono ml-4 shrink-0">
                              Notice Period: <span className="text-white font-bold">{cand.redrob_signals.notice_period_days}d</span> | Offer Acceptance: <span className="text-white font-bold">{cand.redrob_signals.offer_acceptance_rate ? (cand.redrob_signals.offer_acceptance_rate * 100).toFixed(0) : "N/A"}%</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

          </div>

        </div>

      </main>

      <footer className="border-t border-[#21262d] bg-[#161b22] py-4 text-center text-xs text-[#8b949e] mt-auto">
        <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row items-center justify-between">
          <p>© 2026 Recruitment Prioritization Stack. All data simulated under compliance standard gates.</p>
          <div className="flex space-x-4 mt-2 md:mt-0 text-[10px] font-mono">
            <span>Pipeline: Verified</span>
            <span>Auth Scope: Read/Write</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
