import { Link } from "react-router";
import { SiGithub, SiHuggingface } from "react-icons/si";
import { FileText, Database } from "lucide-react";
import { Button } from "./ui/button";

export function Footer() {
  return (
    <footer className="w-full border-t bg-footer-background/70 py-6 md:py-10 flex flex-col items-center">
      <div className="container mx-auto flex flex-col items-center gap-6 px-6 xl:max-w-4xl">
        <div className="flex flex-wrap justify-center gap-3">
          <Button variant="ghost" size="sm" asChild>
            <a
              href="https://arxiv.org/abs/2512.23042"
              target="_blank"
              rel="noopener noreferrer"
            >
              <FileText className="mr-2 h-4 w-4" />
              arXiv
            </a>
          </Button>
          <Button variant="ghost" size="sm" asChild>
            <a
              href="https://github.com/cvpaperchallenge/lam3c"
              target="_blank"
              rel="noopener noreferrer"
            >
              <SiGithub className="mr-2 h-4 w-4" />
              GitHub
            </a>
          </Button>
          <Button variant="ghost" size="sm" asChild>
            <a
              href="https://huggingface.co/aist-cvrt/lam3c-roomtours"
              target="_blank"
              rel="noopener noreferrer"
            >
              <SiHuggingface className="mr-2 h-4 w-4" />
              HuggingFace
            </a>
          </Button>
          <Button variant="ghost" size="sm" disabled>
            <Database className="mr-2 h-4 w-4" />
            Dataset (TBA)
          </Button>
        </div>
        <div className="flex flex-col items-center gap-2 text-center">
          <Link to="/" className="flex items-center space-x-2">
            <span className="font-bold text-lg">LAM3C</span>
          </Link>
          <p className="text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} LAM3C Project. All rights
            reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
