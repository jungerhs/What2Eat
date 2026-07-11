// 应用 Tailwind 配置：保持与原 HTML 等价的 color/font token。
// 由 src/main.js 在 import 时调用。

export function applyTailwindConfig() {
  if (typeof document === 'undefined') return
  if (!document.querySelector('script[data-tailwind-cdn]')) {
    const script = document.createElement('script')
    script.src = 'https://cdn.tailwindcss.com'
    script.dataset.tailwindCdn = '1'
    script.onload = applyConfig
    document.head.appendChild(script)
  } else {
    applyConfig()
  }
}

function applyConfig() {
  // eslint-disable-next-line no-undef
  if (typeof tailwind !== 'undefined') {
    // eslint-disable-next-line no-undef
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'Noto Sans SC', 'system-ui', 'sans-serif'],
            mono: ['JetBrains Mono', 'SF Mono', 'Consolas', 'monospace'],
          },
          colors: {
            brand: {
              50: '#FFF5F0', 100: '#FFE8DD', 200: '#FFCFB8', 300: '#FFB08A',
              400: '#FC8F5C', 500: '#E85D3A', 600: '#D14525', 700: '#B33518',
              800: '#8F2713', 900: '#6B1F12',
            },
            gold: { 400: '#F4C56A', 500: '#E8A838', 600: '#D4911E' },
            warm: {
              50: '#FFFDF9', 100: '#FFF9ED', 200: '#FFF3D6',
              800: '#3D2E1A', 900: '#2B1F0F',
            },
          },
        },
      },
    }
  }
}
