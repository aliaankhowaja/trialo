const DISCLAIMER =
  'This is an educational screening prototype. Outputs are model-generated ' +
  'annotations for teaching and demonstration only — NOT a clinical enrollment ' +
  'decision, not medical advice, and not suitable for patient care or any ' +
  'regulatory purpose. Verify every result against expert judgment.'

export default function DisclaimerBanner() {
  return (
    <div className="disclaimer-banner" role="note">
      <strong>⚠ Educational screening prototype.</strong> {DISCLAIMER}
    </div>
  )
}

export { DISCLAIMER }
