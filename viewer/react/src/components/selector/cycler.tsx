import { url } from "inspector"
import { FunctionComponent, useState } from "react"

interface item {
	name: string
	color: string
	image?: string
	callback: () => void
}

interface CylclerProps {
	items: item[]
	currentItem: number
}

const Cycler: FunctionComponent<CylclerProps> = ({ items, currentItem }) => {
	const [hover, setHover] = useState(false)
	console.log(items[currentItem].image)
	return (
		<>
			<button
				style={{
					borderRadius: "100px",
					//backgroundColor: "white",
					width: "48px",
					height: "48px",
					border: "1px solid white",
					//boxShadow: "0px 0px 8px rgba(34, 34, 34, 0.25)",
					backgroundSize: "contain",
					backgroundImage: items[currentItem].image ? "url(" + items[currentItem].image + ")" : "none",
				}}
				onClick={() => {
					items[currentItem].callback()
				}}
				onMouseEnter={() => {
					setHover(true)
				}}
				onMouseLeave={() => {
					setHover(false)
				}}
			>
				{hover && (
					<div
						style={{
							position: "absolute",
							top: "-30px",

							textAlign: "center",
							background: "rgba(255,255,255,0.2)",
							backdropFilter: "blur(100px)",
							paddingLeft: "8px",
							paddingRight: "8px",
							boxShadow: "0px 0px 8px rgba(34, 34, 34, 0.25)",
							borderRadius: "10px",
						}}
					>
						<p> {items[currentItem].name} </p>
					</div>
				)}
				{!items[currentItem].image && items[currentItem].name}
			</button>
		</>
	)
}

export default Cycler
