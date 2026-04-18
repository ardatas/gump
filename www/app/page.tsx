import dynamic from 'next/dynamic'

const GumpApp = dynamic(() => import('@/components/GumpApp'), { ssr: false })

export default function Home() {
  return <GumpApp />
}
