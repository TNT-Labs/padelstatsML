import { useId, useState, type FC, type ReactNode } from 'react'

/** A section title with an "i" beside it that shows, under the title, what
 *  the section's numbers mean. */
export const InfoHeading: FC<{ title: string; children: ReactNode }> = ({ title, children }) => {
  const [open, setOpen] = useState(false)
  const panel = useId()
  return (
    <>
      <div className="info-heading">
        <h2>{title}</h2>
        <button
          type="button"
          className={`info-btn${open ? ' info-btn-open' : ''}`}
          aria-expanded={open}
          aria-controls={panel}
          aria-label={open ? `Nascondi la spiegazione: ${title}` : `Cosa significano i numeri: ${title}`}
          title={open ? 'Nascondi la spiegazione' : 'Cosa significano questi numeri'}
          onClick={() => setOpen(o => !o)}
        >
          i
        </button>
      </div>
      {open && (
        <div id={panel} className="info-panel">
          {children}
        </div>
      )}
    </>
  )
}
