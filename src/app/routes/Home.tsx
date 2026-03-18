import {
  FileText,
  Copy,
  Check,
  Database,
  Video,
  Network,
  TrendingUp,
} from "lucide-react";
import { SiGithub, SiHuggingface } from "react-icons/si";
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
      <section
        className="relative text-center flex flex-col rounded-3xl overflow-hidden -mx-6 max-w-none"
        style={{
          aspectRatio: "1536 / 1024",
          minHeight: "600px",
          width: "calc(100% + 3rem)",
        }}
      >
        {/* Background Image for Hero - Full height with no opacity */}
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: "url(/lam3c_background.jpg)",
            backgroundSize: "100% 100%",
            backgroundPosition: "center",
            backgroundRepeat: "no-repeat",
            pointerEvents: "none",
          }}
        />
        {/* Opacity overlay - bottom half with gradient */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "linear-gradient(to bottom, transparent 40%, rgba(0, 0, 0, 0.6) 70%, rgba(0, 0, 0, 0.8) 100%)",
            pointerEvents: "none",
          }}
        />
        <div className="relative z-10 flex flex-col h-full justify-between py-6 sm:py-8">
          {/* Venue - stays at top */}
          <div className="flex-shrink-0">
            <div className="inline-block rounded-full border bg-card px-4 py-1.5 text-sm font-medium text-muted-foreground">
              {lam3cData.venue}
            </div>
          </div>

          {/* Subtitle and content - centered and pushed to bottom half */}
          <div className="space-y-3 flex-shrink-0 mt-auto">
            {/* <img
              src="/logo.png"
              alt="LAM3C"
              className="mx-auto h-32 w-auto sm:h-44 md:h-56"
            /> */}
            {/* Subtitle */}
            <p className="text-lg text-white sm:text-xl md:text-2xl lg:text-3xl tracking-tight font-semibold px-4">
              {lam3cData.subtitle}
            </p>

            {/* Authors */}
            <div className="space-y-2">
              <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 text-sm sm:text-base text-white px-2">
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
                    <sup className="text-xs text-gray-300 ml-0.5">
                      {author.affiliationIds.join(",")}
                    </sup>
                  </span>
                ))}
              </div>
              <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 text-xs sm:text-sm text-gray-300 px-2">
                {lam3cData.affiliations.map((aff) => (
                  <span key={aff.id}>
                    <sup className="mr-0.5">{aff.id}</sup>
                    {aff.name}
                  </span>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-wrap justify-center gap-2 sm:gap-3 px-2">
              <Button variant="outline" asChild>
                <a
                  href={lam3cData.links.arxiv}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <FileText className="mr-2 h-4 w-4" />
                  Paper
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
                  <SiHuggingface className="mr-2 h-4 w-4" />
                  Models
                </a>
              </Button>
              <Button variant="outline" disabled>
                <Database className="mr-2 h-4 w-4" />
                Dataset (TBA)
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Teaser Image */}
      <div className="mx-auto max-w-3xl">
        <img
          src="/lam3c_scaling.png"
          alt="LAM3C scaling results showing performance improvement with more data"
          className="w-full rounded-xl border shadow-sm"
          loading="lazy"
        />
      </div>

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
        <p className="text-lg italic leading-relaxed mb-2">LAM3C Message</p>
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
        <div className="grid gap-6 md:grid-cols-3">
          {lam3cData.contributions.map((c, i) => {
            const icons = [Video, Network, TrendingUp];
            const Icon = icons[i];
            return (
              <div key={i} className="rounded-xl border bg-card p-6 space-y-3">
                <div className="flex items-center gap-3">
                  <div className="rounded-lg bg-primary/10 p-2.5">
                    <Icon className="h-5 w-5 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold">{c.name}</h3>
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {c.description}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* RoomTours Gallery */}
      <section id="results" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          RoomTours Gallery
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

      {/* LAM3C Pipeline */}
      <section id="pipeline" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          LAM3C Pipeline
        </h2>
        <p className="text-muted-foreground">
          LAM3C employs a noise-regularized self-supervised learning framework
          that learns robust 3D representations from noisy video-generated point
          clouds through Laplacian smoothing and student-teacher distillation.
        </p>

        {/* Embedding Stability with Laplacian Smoothing */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold">
            Embedding Stability with Laplacian Smoothing
          </h3>
          <div className="mx-auto max-w-3xl">
            <img
              src="/lam3c_fig2.jpg"
              alt="LAM3C embedding stability through Laplacian smoothing"
              className="w-full rounded-xl border shadow-sm"
              loading="lazy"
            />
          </div>
          <div className="space-y-3 text-muted-foreground">
            <p>
              <strong className="text-foreground">
                Laplacian Smoothing Loss (ℒ<sub>smooth</sub>):
              </strong>{" "}
              Video-generated point clouds contain inherent reconstruction noise
              that can destabilize representation learning. To address this, we
              introduce a Laplacian smoothing regularization that enforces local
              geometric consistency in the embedding space. This loss ensures
              that neighboring points in the k-NN graph have similar embeddings,
              making the learned representations robust to point-level noise
              while preserving semantic structure.
            </p>
            <p>
              The smoothing loss operates by constructing a k-nearest neighbor
              graph from the noisy input and penalizing large embedding
              differences between connected points. This encourages the encoder
              to learn stable features that capture geometric structure rather
              than fitting to noise artifacts. The arrow width in the
              visualization indicates attraction strength, where thicker arrows
              represent stronger geometric consistency constraints between
              neighboring points.
            </p>
          </div>
        </div>

        {/* Student-Teacher Framework */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold">
            Student-Teacher Knowledge Distillation
          </h3>
          <div className="mx-auto max-w-3xl">
            <img
              src="/lam3c_fig3.jpg"
              alt="LAM3C student-teacher framework with masked learning"
              className="w-full rounded-xl border shadow-sm"
              loading="lazy"
            />
          </div>
          <div className="space-y-3 text-muted-foreground">
            <p>
              <strong className="text-foreground">
                Consistency Loss (ℒ<sub>consist</sub>):
              </strong>{" "}
              LAM3C employs a student-teacher framework where the student
              network learns from both masked reconstruction and consistency
              with the teacher&apos;s predictions. The student receives masked
              point cloud patches from global views, while the teacher processes
              the complete scene with noise augmentation. This asymmetric
              learning setup encourages the student to develop robust
              representations that generalize across different noise conditions
              and missing regions.
            </p>
            <p>
              <strong className="text-foreground">
                Distillation Loss (ℒ<sub>distill</sub>):
              </strong>{" "}
              The teacher network, updated via exponential moving average (EMA)
              of the student weights, provides stable learning targets. The
              distillation loss minimizes the divergence between student and
              teacher embeddings, ensuring that the learned representations
              remain consistent despite input perturbations. The &quot;Pull
              together&quot; mechanism shown in the figure illustrates how
              embeddings from the masked student view and augmented teacher view
              are aligned in the feature space, promoting view-invariant feature
              learning.
            </p>
            <p>
              <strong className="text-foreground">
                Combined Training Objective:
              </strong>{" "}
              The overall LAM3C loss combines these components: ℒ = ℒ
              <sub>smooth</sub> + λ<sub>c</sub>ℒ<sub>consist</sub> + λ
              <sub>d</sub>ℒ<sub>distill</sub>, where the smoothing term handles
              point-level noise, while the consistency and distillation terms
              enable robust learning from incomplete and noisy observations.
              This multi-faceted approach allows LAM3C to effectively pre-train
              on large-scale video-generated point clouds without requiring
              clean 3D scans.
            </p>
          </div>
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

      {/* Qualitative Results */}
      <section id="qualitative" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Qualitative Results
        </h2>
        <div className="mx-auto max-w-3xl">
          <img
            src="/lam3c_demo.png"
            alt="Zero-shot semantic visualization comparison between Sonata and LAM3C"
            className="w-full rounded-xl border shadow-sm"
            loading="lazy"
          />
        </div>
        <p className="text-muted-foreground leading-relaxed">
          The figure above demonstrates LAM3C&apos;s learned representations
          through PCA-based visualization, comparing results with Sonata
          (trained on real 3D scans) and LAM3C (trained on video-generated
          RoomTours data). Without any task-specific fine-tuning, LAM3C shows
          clear segmentation of local structures such as desks, chairs, and
          walls, indicating that the model successfully learns meaningful
          geometric representations directly from noisy video-reconstructed
          point clouds. While LAM3C demonstrates strong local structure
          understanding comparable to models trained on clean 3D scans, the
          visualization reveals slightly less coherent global structure. This
          trade-off stems from the inherent characteristics of video-generated
          point clouds, where variations in coordinate frames and scale make
          learning globally consistent scene geometry more challenging.
          Nevertheless, LAM3C successfully demonstrates that large-scale
          pre-training on video-generated data can produce robust 3D
          representations without requiring expensive 3D scanning equipment.
        </p>
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
