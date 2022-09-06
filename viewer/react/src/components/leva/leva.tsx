import { useAvatars } from "@/stores/avatarStore"
import { button, folder, useControls } from "leva"
import { FunctionComponent } from "react"

interface LevaProps {}

const Leva: FunctionComponent<LevaProps> = ({}) => {
	const reductionPercentage = useAvatars((state) => state.reductionPercentage)
	const avatarNames = useAvatars((state) => state.avatarNames)
	const environmentNames = useAvatars((state) => state.environmentNames)

	useControls(
		{
			"Avatar Loader": folder({
				Name: {
					options: avatarNames ? avatarNames : [],
					onChange: (name) => {
						useAvatars.getState().actions.setAvatarName(name)
						useAvatars.getState().actions.loadTrueAvatar(undefined)
					},
				},
			}),
			"Avatar Estimator": folder({
				Environment: {
					options: environmentNames ? environmentNames : [],
					onChange: (name) => {
						useAvatars.getState().actions.setEnvironmentName(name)
					},
				},
				Estimate: button(() => useAvatars.getState().actions.loadEstimatedAvatar(undefined)),
			}),
			Settings: folder({
				Density: {
					value: reductionPercentage,
					min: 0,
					max: 1,
					step: 0.01,
					onEditEnd(value, path, context) {
						console.log(value)
						useAvatars.getState().actions.setReductionPercentage(value)
					},
				},
			}),
		},
		[avatarNames, environmentNames]
	)

	return <></>
}

export default Leva
