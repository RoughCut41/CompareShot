/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Custom dark palette beyond zinc defaults
        background: '#09090b',
        surface: '#18181b',
        'surface-alt': '#1c1c1e',
      },
    },
  },
  plugins: [],
};
