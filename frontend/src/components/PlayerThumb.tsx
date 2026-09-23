import { useState, type FC } from 'react'
import { apiAsset } from '../lib/format'

const BASE = import.meta.env.VITE_API_URL ?? ''

interface Props {
  url?: string | null
  alt: string
  /** 'player-thumb' in the stats cards, 'identify-thumb' when naming players. */
  className: string
}

/** A player's picture, or the placeholder when there is none — including
 *  when the file fails to load, instead of a broken-image icon. */
export const PlayerThumb: FC<Props> = ({ url, alt, className }) => {
  const [failed, setFailed] = useState(false)
  if (!url || failed) {
    return <div className={`${className} ${className}-empty`}>👤</div>
  }
  return <img src={apiAsset(url, BASE)} alt={alt} className={className} onError={() => setFailed(true)} />
}
