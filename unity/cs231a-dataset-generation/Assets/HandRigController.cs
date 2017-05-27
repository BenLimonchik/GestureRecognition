using System;
using System.Reflection;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class HandRigController : MonoBehaviour
{
	[System.Serializable]
	public class JointAngle
	{
		public Transform Transform;
		public Vector3 InitialAngles;
		public Vector3 Angles;
	}

	[System.Serializable]
	public class Palm
	{
		[System.Serializable]
		public class Thumb
		{
			public JointAngle Palm;
			public JointAngle Joint1;
			public JointAngle Joint2;
		}

		public Thumb thumb;
	}

	public Palm palm;

	private JointAngle[] joints;

	private void Start ()
	{
		Debug.Log("Saving initial angles of joints.");
		ForEachJointAngle((JointAngle joint) => {
			joint.InitialAngles = joint.Transform.localEulerAngles;
			joint.Angles = joint.InitialAngles;
		});
	}

	public void Update ()
	{
		Debug.Log("Updating local euler angles.");
		ForEachJointAngle((JointAngle joint) => {
			joint.Transform.localEulerAngles = joint.Angles;
		});
	}

	private void ForEachJointAngle(Action<JointAngle> lambda) {
		foreach (var joint in new JointAngle[]{
			palm.thumb.Palm,
			palm.thumb.Joint1,
			palm.thumb.Joint2 
		})
		{
			lambda(joint);
		}
	}
}
