import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://locobench.org',
  base: '/docs',
  integrations: [
    starlight({
      title: 'LocoBench',
      description: 'I have X GB of VRAM — what\'s the best model I can run?',
      social: [
        { icon: 'external', label: 'Home', href: 'https://locobench.org' },
        { icon: 'external', label: 'LocoLab', href: 'https://locolabo.org' },
      ],
      customCss: ['./src/styles/custom.css'],
      head: [
        { tag: 'script', attrs: { src: 'https://cdn.plot.ly/plotly-2.35.2.min.js', defer: true } },
        { tag: 'script', attrs: { src: '/docs/javascripts/benchmark_data.js', defer: true } },
        { tag: 'script', attrs: { src: '/docs/javascripts/chart_theme.js', defer: true } },
        { tag: 'script', attrs: { src: '/docs/javascripts/charts_overview.js', defer: true } },
        { tag: 'script', attrs: { src: '/docs/javascripts/charts_quality.js', defer: true } },
        { tag: 'script', attrs: { src: '/docs/javascripts/charts_speed.js', defer: true } },
        { tag: 'script', attrs: { src: '/docs/javascripts/charts_efficiency.js', defer: true } },
      ],
      sidebar: [
        {
          label: 'Results',
          items: [
            { label: 'Quality Analysis', slug: 'quality' },
            { label: 'Speed Analysis', slug: 'speed' },
            { label: 'Bang per Bit', slug: 'bang-per-bit' },
          ],
        },
        {
          label: 'Hardware',
          items: [
            { label: 'Colmena', slug: 'colmena' },
            { label: 'VRAM Tiers', slug: 'tiers' },
          ],
        },
        {
          label: 'Guide',
          items: [
            { label: 'Benchmarking Guide', slug: 'guide' },
          ],
        },
      ],
    }),
  ],
});
