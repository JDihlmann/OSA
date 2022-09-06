import { extend, MaterialNode, Object3DNode } from "@react-three/fiber"
import { ShaderLib, ShaderMaterial, Texture, Vector2, Vector3 } from "three"

const vertexShaderText = /* glsl */ `
    void main() {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`

const fragmentShaderText = /* glsl */ `
    void main() {
		gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
    }
`

export class DotMaterial extends ShaderMaterial {
	constructor() {
		super({
			transparent: false,
			uniforms: { size: { value: 2 } },
			vertexShader: ShaderLib.points.vertexShader,
			fragmentShader: /* glsl */ `
				varying vec3 vColor;
				void main() {
					gl_FragColor = vec4(vColor, 1.0);
				}`,
		})
	}
}

extend({ DotMaterial })

declare global {
	namespace JSX {
		interface IntrinsicElements {
			dotMaterial: MaterialNode<DotMaterial, typeof DotMaterial>
		}
	}
}
