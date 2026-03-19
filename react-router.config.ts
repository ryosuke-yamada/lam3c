import type { Config } from "@react-router/dev/config";

export default {
  // Config options...
  appDirectory: "src/app",
  // Server-side render by default, to enable SPA mode set this to `false`
  ssr: false,
  basename: process.env.NODE_ENV === 'production' ? '/lam3c' : undefined,
  // async prerender() {
  //   return ["/popular"];
  // },
} satisfies Config;