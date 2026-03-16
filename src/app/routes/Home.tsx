import { ExternalLink, Copy, Check } from "lucide-react";
import { SiGithub } from "react-icons/si";
import { useLocation } from "react-router";
import { useEffect, useState, useCallback } from "react";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../components/ui/table";
import { Button } from "../../components/ui/button";
import lam3cData from "../../data/lam3c.json";
import type { Route } from "./+types/Home";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { buildMeta } from "@/lib/seo";

export const meta: Route.MetaFunction = () =>
  buildMeta({
    title:
      "LAM3C: 3D sans 3D Scans — Scalable Pre-training from Video-Generated Point Clouds",
    description: lam3cData.abstract,
    path: "/",
    keywords: [
      "LAM3C",
      "3D self-supervised learning",
      "point cloud",
      "video-generated",
      "RoomTours",
      "CVPR 2026",
    ],
  });

function CopyBibtexButton() {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(lam3cData.bibtex).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, []);

  return (
    <Button variant="outline" size="sm" onClick={handleCopy}>
      {copied ? (
        <>
          <Check className="mr-2 h-4 w-4" />
          Copied
        </>
      ) : (
        <>
          <Copy className="mr-2 h-4 w-4" />
          Copy BibTeX
        </>
      )}
    </Button>
  );
}

function Home() {
  const location = useLocation();

  useEffect(() => {
    if (!location.hash) return;
    const element = document.querySelector(location.hash);
    element?.scrollIntoView({ behavior: "smooth" });
  }, [location.hash]);

  return (
    <main className="container px-6 py-8 space-y-20 xl:w-4xl">
      {/* Hero Section */}
      <section className="text-center space-y-8 pt-8">
        <div className="inline-block rounded-full border bg-card px-4 py-1.5 text-sm font-medium text-muted-foreground">
          {lam3cData.venue}
        </div>
        <div className="space-y-4">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">
            {lam3cData.title}
          </h1>
          <p className="text-xl text-muted-foreground sm:text-2xl md:text-3xl tracking-tight">
            {lam3cData.subtitle}
          </p>
        </div>

        {/* Authors */}
        <div className="space-y-3">
          <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-base">
            {lam3cData.authors.map((author, i) => (
              <span key={i} className="whitespace-nowrap">
                {author.url ? (
                  <a
                    href={author.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-primary"
                  >
                    {author.name}
                  </a>
                ) : (
                  author.name
                )}
                <sup className="text-xs text-muted-foreground ml-0.5">
                  {author.affiliationIds.join(",")}
                </sup>
              </span>
            ))}
          </div>
          <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-sm text-muted-foreground">
            {lam3cData.affiliations.map((aff) => (
              <span key={aff.id}>
                <sup className="mr-0.5">{aff.id}</sup>
                {aff.name}
              </span>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap justify-center gap-3">
          <Button variant="outline" asChild>
            <a
              href={lam3cData.links.arxiv}
              target="_blank"
              rel="noopener noreferrer"
            >
              <ExternalLink className="mr-2 h-4 w-4" />
              Paper (arXiv)
            </a>
          </Button>
          <Button variant="outline" asChild>
            <a
              href={lam3cData.links.github}
              target="_blank"
              rel="noopener noreferrer"
            >
              <SiGithub className="mr-2 h-4 w-4" />
              Code
            </a>
          </Button>
          <Button variant="outline" asChild>
            <a
              href={lam3cData.links.huggingface}
              target="_blank"
              rel="noopener noreferrer"
            >
              Models (HuggingFace)
            </a>
          </Button>
        </div>

        {/* Teaser Image */}
        <div className="mx-auto max-w-3xl">
          <img
            src="/lam3c_demo.png"
            alt="LAM3C PCA visualization and similarity heatmap demo"
            className="w-full rounded-xl border shadow-sm"
            loading="lazy"
          />
        </div>
      </section>

      {/* Abstract */}
      <section id="abstract" className="space-y-4">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Abstract
        </h2>
        <p className="leading-relaxed text-muted-foreground">
          {lam3cData.abstract}
        </p>
      </section>

      {/* Key Message */}
      <section className="rounded-xl border bg-card p-8">
        <blockquote className="space-y-4">
          <p className="text-lg italic leading-relaxed">
            &ldquo;{lam3cData.message}&rdquo;
          </p>
        </blockquote>
      </section>

      {/* Contributions */}
      <section id="method" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Key Contributions
        </h2>
        <div className="grid gap-6 md:grid-cols-2">
          {lam3cData.contributions.map((c, i) => (
            <div key={i} className="rounded-xl border bg-card p-6 space-y-2">
              <h3 className="text-xl font-semibold">{c.name}</h3>
              <p className="text-muted-foreground">{c.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Scaling Results */}
      <section id="results" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Scaling Results
        </h2>
        <div className="mx-auto max-w-3xl">
          <img
            src="/lam3c_scaling.png"
            alt="LAM3C scaling results showing performance improvement with more data"
            className="w-full rounded-xl border shadow-sm"
            loading="lazy"
          />
        </div>
      </section>

      {/* Benchmark Results Table */}
      <section id="benchmarks" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Benchmark Results
        </h2>
        <p className="text-muted-foreground">{lam3cData.results.description}</p>
        <ScrollArea className="w-full">
          <Table>
            <TableHeader>
              <TableRow>
                {lam3cData.results.table.headers.map((h, i) => (
                  <TableHead key={i} className={i > 1 ? "text-center" : ""}>
                    {h}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {lam3cData.results.table.rows.map((row, i) => (
                <TableRow key={i}>
                  {row.map((cell, j) => (
                    <TableCell
                      key={j}
                      className={j > 1 ? "text-center" : "font-medium"}
                    >
                      {cell}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
        <p className="text-sm text-muted-foreground">
          LP = Linear Probing, FT = Full Fine-tuning. All numbers are mIoU (%).
        </p>
      </section>

      {/* RoomTours Pipeline */}
      <section id="pipeline" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          RoomTours Pipeline
        </h2>
        <p className="text-muted-foreground">
          RoomTours converts unlabeled indoor videos into training-ready point
          clouds for LAM3C pre-training. The pipeline includes video download,
          scene segmentation, Pi3 reconstruction, and point-cloud preprocessing.
        </p>
        <div className="mx-auto max-w-3xl">
          <img
            src="/roomtours.png"
            alt="RoomTours pipeline overview: from internet videos to 3D point clouds"
            className="w-full rounded-xl border shadow-sm"
            loading="lazy"
          />
        </div>
      </section>

      {/* News */}
      <section id="news" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">News</h2>
        <div className="space-y-3">
          {lam3cData.news.map((item, i) => (
            <div key={i} className="flex gap-4">
              <span className="shrink-0 font-medium text-muted-foreground w-24">
                {item.date}
              </span>
              <span>{item.content}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Citation */}
      <section id="citation" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Citation
        </h2>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              If you find our work useful, please cite:
            </p>
            <CopyBibtexButton />
          </div>
          <pre className="overflow-x-auto rounded-xl border bg-card p-4 text-sm">
            <code>{lam3cData.bibtex}</code>
          </pre>
        </div>
      </section>
    </main>
  );
}

export default Home;
