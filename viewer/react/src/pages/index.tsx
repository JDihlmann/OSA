import Leva from "@/components/leva/leva"
import Scene from "@/components/scenes/scene"
import Selector from "@/components/selector/selector"

const Home = () => {
	return (
		<div style={{ position: "absolute", width: "100%", height: "100%" }}>
			<Scene />
			<Leva />
			<Selector />
		</div>
	)
}

export default Home
