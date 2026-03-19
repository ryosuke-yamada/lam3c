import type { MetaDescriptor } from "react-router";

const SITE_URL = "https://cvpaperchallenge.github.io/lam3c";
const DEFAULT_IMAGE = `${SITE_URL}/lam3c_ogp.jpg`;
const DEFAULT_IMAGE_ALT =
  "LAM3C: PCA visualization and similarity heatmap of 3D point cloud features";
const SITE_NAME =
  "LAM3C: 3D sans 3D Scans — Scalable Pre-training from Video-Generated Point Clouds";
const DEFAULT_DESCRIPTION =
  "LAM3C learns 3D representations from video-generated point clouds without real 3D scans. Accepted to CVPR 2026.";
const DEFAULT_KEYWORDS = [
  "LAM3C",
  "3D self-supervised learning",
  "point cloud",
  "video-generated point clouds",
  "RoomTours",
  "CVPR 2026",
  "semantic segmentation",
  "instance segmentation",
];

type SeoConfig = {
  title?: string;
  description?: string;
  path?: string;
  image?: string;
  imageAlt?: string;
  type?: string;
  keywords?: string[];
};

const normalizePath = (path?: string) => {
  if (!path || path === "/") {
    return "";
  }

  return path.startsWith("/") ? path : `/${path}`;
};

export function buildMeta(config: SeoConfig = {}): MetaDescriptor[] {
  const {
    title = SITE_NAME,
    description = DEFAULT_DESCRIPTION,
    path,
    image = DEFAULT_IMAGE,
    imageAlt = DEFAULT_IMAGE_ALT,
    type = "website",
    keywords = [],
  } = config;

  const canonicalPath = normalizePath(path);
  const url = `${SITE_URL}${canonicalPath}`;
  const keywordSet = new Set([
    ...DEFAULT_KEYWORDS,
    ...keywords.filter((keyword) => keyword.trim().length > 0),
  ]);
  const keywordContent = Array.from(keywordSet).join(", ");

  const descriptors: MetaDescriptor[] = [
    { title },
    { name: "description", content: description },
    ...(keywordContent ? [{ name: "keywords", content: keywordContent }] : []),
    { property: "og:type", content: type },
    { property: "og:site_name", content: SITE_NAME },
    { property: "og:title", content: title },
    { property: "og:description", content: description },
    { property: "og:url", content: url },
    { property: "og:image", content: image },
    { property: "og:image:alt", content: imageAlt },
    { name: "twitter:card", content: "summary_large_image" },
    { name: "twitter:title", content: title },
    { name: "twitter:description", content: description },
    { name: "twitter:image", content: image },
    { name: "twitter:image:alt", content: imageAlt },
    { tagName: "link", rel: "canonical", href: url },
  ];

  return descriptors;
}

export const seoDefaults = {
  SITE_NAME,
  SITE_URL,
  DEFAULT_DESCRIPTION,
  DEFAULT_IMAGE,
  DEFAULT_IMAGE_ALT,
  DEFAULT_KEYWORDS,
};
