import { DistanceType, SelectorMode, ContentType, SurfaceType, useSelector } from "@/stores/selectorStore"
import { FunctionComponent, useState } from "react"
import Cycler from "./cycler"

interface SelectorProps {}

const Selector: FunctionComponent<SelectorProps> = ({}) => {
	const mode = useSelector((state) => state.mode)
	const distanceType = useSelector((state) => state.distanceType)
	const contentType = useSelector((state) => state.contentType)
	const surfaceType = useSelector((state) => state.surfaceType)

	const setMode = useSelector((state) => state.actions.setMode)
	const setDistanceType = useSelector((state) => state.actions.setDistanceType)
	const setContentType = useSelector((state) => state.actions.setContentType)
	const setSurfaceType = useSelector((state) => state.actions.setSurfaceType)

	return (
		<div
			style={{
				position: "absolute",
				height: "100px",
				width: "100%",
				bottom: "0",
				left: "0",
				pointerEvents: "none",
			}}
		>
			<div
				style={{
					display: "flex",
					flexBasis: "auto",
					justifyContent: "center",
					alignItems: "center",
					height: "100%",
					pointerEvents: "none",
				}}
			>
				<div
					style={{
						display: "flex",
						flexBasis: "auto",
						justifyContent: "center",
						alignItems: "center",
						height: "60px",
						background: "rgba(255,255,255,0.2)",
						backdropFilter: "blur(20px)",
						gap: "16px",
						paddingLeft: "8px",
						paddingRight: "8px",
						boxShadow: "0px 0px 8px rgba(34, 34, 34, 0.25)",
						pointerEvents: "auto",
						borderRadius: "100px",
					}}
				>
					<Cycler
						currentItem={mode}
						items={[
							{
								name: "True",
								color: "white",
								image: "images/true.png",
								callback: () => {
									setMode(SelectorMode.Estimated)
								},
							},
							{
								name: "Estimated",
								color: "white",
								image: "images/estimate.png",
								callback: () => {
									setMode(SelectorMode.Both)
								},
							},
							{
								name: "Both",
								color: "white",
								image: "images/both.png",
								callback: () => {
									setMode(SelectorMode.True)
								},
							},
						]}
					/>
					<div style={{ width: "40px" }} />
					<Cycler
						currentItem={distanceType}
						items={[
							{
								name: "Far",
								color: "white",
								image: "images/far.png",
								callback: () => {
									setDistanceType(DistanceType.Near)
								},
							},
							{
								name: "Near",
								color: "white",
								image: "images/near.png",
								callback: () => {
									setDistanceType(DistanceType.Far)
								},
							},
						]}
					/>
					<Cycler
						currentItem={contentType}
						items={[
							{
								name: "Color",
								color: "white",
								image: "images/color.png",
								callback: () => {
									setContentType(ContentType.Normal)
								},
							},
							{
								name: "Normal",
								color: "white",
								image: "images/normal.png",
								callback: () => {
									setContentType(ContentType.Inside)
								},
							},
							{
								name: "Inside",
								color: "white",
								image: "images/inside.png",
								callback: () => {
									setContentType(ContentType.Outside)
								},
							},
							{
								name: "Outside",
								color: "white",
								image: "images/outside.png",
								callback: () => {
									setContentType(ContentType.Color)
								},
							},
						]}
					/>
					<Cycler
						currentItem={surfaceType}
						items={[
							{
								name: "Default",
								color: "white",
								image: "images/default.png",
								callback: () => {
									setSurfaceType(SurfaceType.Project)
								},
							},
							{
								name: "Project",
								color: "white",
								image: "images/projected.png",
								callback: () => {
									setSurfaceType(SurfaceType.Default)
								},
							},
						]}
					/>
				</div>
			</div>
		</div>
	)
}

export default Selector
