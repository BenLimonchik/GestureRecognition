using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ImageCaptureController : MonoBehaviour {
	public enum Type {
		Complex, Uniform
	}

	public enum Gesture {
		A, B, C, Five, Point, Rock, ThumbsUp, V
	}

	private Dictionary<Gesture, string> prefabs = new Dictionary<Gesture, string>();
	public Type imageType;
	public Gesture gesture;

	public int numScreenshots = 0;
	public int startIndex;
	public string fileName = "derp";
	public string fileExtension = "jpg";

	public MeshRenderer bgRenderer;
	public Material bgComplexMtl;
	public Material bgUniformMtl;
	public int numBackgrounds = 1;

	public Light sunLight;
	public float maxSunRotX, maxSunRotY, maxSunRotZ;
	public float lightIntensityMin = 0.0f;
	public float lightIntensityMax = 1.0f;

	private Transform hand;
	private HandRigController handRig;
	private SkinnedMeshRenderer handRenderer;
	public Material[] handMaterials;
	public float maxDeltaX, maxDeltaY, maxDeltaZ, maxRotX, maxRotY, maxRotZ, minScale, maxScale;
	public float maxHandRigRotX, maxHandRigRotY, maxHandRigRotZ;

	private Vector3 sunInitialRotation;
	private Vector3 handInitialPosition;
	private Vector3 handInitialRotation;

	private string outputName = "";
	private int i = 0;
	private Color averageBgColor = Color.black;
	private int[] backgroundOrder;

	void Start() {
		prefabs[Gesture.A] = "Hands/Hand Rig - A";
		prefabs[Gesture.B] = "Hands/Hand Rig - B";
		prefabs[Gesture.C] = "Hands/Hand Rig - C";
		prefabs[Gesture.Five] = "Hands/Hand Rig - Five";
		prefabs[Gesture.Point] = "Hands/Hand Rig - Point";
		prefabs[Gesture.Rock] = "Hands/Hand Rig - Rock";
		prefabs[Gesture.ThumbsUp] = "Hands/Hand Rig - ThumbsUp";
		prefabs[Gesture.V] = "Hands/Hand Rig - V";

		var handObject = GameObject.Instantiate(Resources.Load(prefabs[gesture])) as GameObject;
		hand = handObject.transform;
		handRig = handObject.GetComponentInChildren<HandRigController>();
		handRenderer = handObject.GetComponentInChildren<SkinnedMeshRenderer>();

		int ticks = System.Environment.TickCount;
		Debug.Log("Ticks => seed = " + ticks);
		Random.seed = ticks;

		sunInitialRotation = sunLight.transform.eulerAngles;
		handInitialPosition = hand.position;
		handInitialRotation = hand.eulerAngles;

		backgroundOrder = new int[numBackgrounds];
		StringBuilder sb = new StringBuilder();
		sb.Append("[");
		for (int j = 0; j < numBackgrounds; ++j) {
			int numBgsLeft = numBackgrounds - j;
			backgroundOrder[j] = Random.Range(1, numBgsLeft);
			sb.Append(backgroundOrder[j].ToString());
			sb.Append(",");
		}
		sb.Append("]");
		Debug.Log(sb.ToString());
	}

	void OnPreRender() {
		if (i < numScreenshots) {
			SetupBackground();
			SetupLighting();
			SetupHandModel();
			outputName = string.Format("{0}{1}.{2}", fileName, (i + 1 + startIndex).ToString("D4"), fileExtension);
			++i;
		}
	}

	void OnPostRender() {
		if (outputName.Length > 0) {
			//Application.CaptureScreenshot(outputName);
			int targetWidth = 66;
			int targetHeight = 76;

			var shot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
			shot.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0, false);
			while (shot.width > 2 * targetWidth) {
				TextureScale.Bilinear(shot, shot.width / 2, shot.height / 2);
			}
			TextureScale.Bilinear(shot, 66, 76);
			if (shot.width > 66 || shot.height > 76) {
				//TextureScale.Bilinear(shot, 66, 76);
			}
			var bytes = shot.EncodeToPNG();
			FileStream file = File.Open(outputName, System.IO.FileMode.Create);
			BinaryWriter binaryWriter = new BinaryWriter(file);
			binaryWriter.Write(bytes);
			file.Close();

			outputName = "";
		}
	}

	private void SetupBackground() {
		if (imageType == Type.Complex) {
			bgRenderer.material = bgComplexMtl;

			string texturePath = string.Format("Backgrounds/pic_{0}", backgroundOrder[i % numBackgrounds].ToString("D3"));
			Debug.Log(texturePath);
			Texture2D texture = Resources.Load(texturePath) as Texture2D;
			//averageBgColor = GetAverageColor(texture);
			bgComplexMtl.mainTexture = texture;
			bgComplexMtl.mainTextureOffset = new Vector2(Random.value, Random.value);
		}

		else if (imageType == Type.Uniform) {
			bgRenderer.material = bgUniformMtl;
			bgUniformMtl.color = new Color(Random.Range(0.75f, 0.85f), Random.Range(0.75f, 0.85f), Random.Range(0.75f, 0.85f));
		}
	}

	private void SetupLighting() {
		float rotX = Random.Range(-maxRotX, maxRotX);
		float rotY = Random.Range(-maxRotX, maxRotX);
		float rotZ = Random.Range(-maxRotX, maxRotX);

		sunLight.transform.eulerAngles = sunInitialRotation + new Vector3(rotX, rotY, rotZ);

		Debug.Assert(lightIntensityMin < lightIntensityMax);
		sunLight.intensity = (lightIntensityMax - lightIntensityMin) * Random.value + lightIntensityMin;
		//sunLight.color = averageBgColor;

		//Debug.LogFormat("Average BG color [{0}, {1}, {2}]", averageBgColor.r, averageBgColor.g, averageBgColor.b);
	}

	private void RandomizeJointAngle(HandRigController.JointAngle joint) {
		joint.Angles = joint.InitialAngles + new Vector3(
			Random.Range(-maxHandRigRotX, maxHandRigRotX),
			Random.Range(-maxHandRigRotY, maxHandRigRotY),
			Random.Range(-maxHandRigRotZ, maxHandRigRotZ));
	}

	private void SetupHandModel() {
		float deltaX = Random.Range(-maxDeltaX, maxDeltaX);
		float deltaY = Random.Range(-maxDeltaY, maxDeltaY);
		float deltaZ = Random.Range(-maxDeltaZ, maxDeltaZ);
		float rotX = Random.Range(-maxRotX, maxRotX);
		float rotY = Random.Range(-maxRotX, maxRotX);
		float rotZ = Random.Range(-maxRotX, maxRotX);
		float scale = Random.Range(minScale, maxScale);

		hand.position = handInitialPosition + new Vector3(deltaX, deltaY, deltaZ);
		hand.eulerAngles = handInitialRotation + new Vector3(rotX, rotY, rotZ);
		hand.localScale = new Vector3(scale, scale, scale);

		Material mat = handMaterials[Random.Range(0, handMaterials.Length)];
		handRenderer.material = mat;

		handRig.ForEachJointAngle(RandomizeJointAngle);
	}

	private Color GetAverageColor(Texture2D tex) {
		Color[] pixels = tex.GetPixels();

		Color sum = new Color();
		for (int i = 0; i < pixels.Length; ++i) {
			sum += pixels[i];
		}

		sum /= pixels.Length;
		return sum;
	}
}
