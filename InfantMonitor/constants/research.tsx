import { StaticImageData } from "next/image";
import AVSuperb_img from "@/static/research/av-superb.webp"
import DBC_img from "@/static/research/dbc.webp"
import CUDA_img from "@/static/research/cuda-dst.webp"

export type ResearchType = {
  image: StaticImageData,
  title: string,
  authors: {
    name: string,
    link?: string,
    me?: boolean,
  }[],
  link: string,
  journal: string,
  year: string,
  description: string,
  others: {
    "Project Page"?: string,
    arXiv?: string,
    Video?: string,
    Code?: string,
  },
}

type ResearchDataType = {
  interest: string[],
  researchList: ResearchType[],
}

const researchData: ResearchDataType = {
  interest: [
    "I have a keen interest in exploring various aspects of machine learning, such as natural language processing and reinforcement learning. Representative papers are highlighted.",
  ],
  researchList: [
    {
      image: AVSuperb_img,
      title: "AV-SUPERB: A Multi-Task Evaluation Benchmark for Audio-Visual Representation Models",
      link: "https://av.superbbenchmark.org/",
      authors: [
        // Yuan Tseng, Layne Berry, Yi-Ting Chen, I-Hsiang Chiu, Hsuan-Hao Lin, Max Liu, Puyuan Peng, Yi-Jen Shih, Hung-Yu Wang, Haibin Wu, Po-Yao Huang, Chun-Mao Lai, Shang-Wen Li, David Harwath, Yu Tsao, Shinji Watanabe, Abdelrahman Mohamed, Chi-Luen Feng, Hung-yi Lee
        {
          name: "Yuan Tseng",
        },
        {
          name: "Layne Berry*",
        },
        {
          name: "Yi-Ting Chen*",
        },
        {
          name: "I-Hsiang Chiu*",
        },
        {
          name: "Hsuan-Hao Lin*",
        },
        {
          name: "Max Liu*",
        },
        {
          name: "Puyuan Peng*",
        },
        {
          name: "Yi-Jen Shih*",
        },
        {
          name: "Hung-Yu Wang*",
        },
        {
          name: "Haibin Wu*",
        },
        {
          name: "Po-Yao Huang",
        },
        {
          name: "Chun-Mao Lai",
          link: "https://www.mecoli.net/",
          me: true,
        },
        {
          name: "Shang-Wen Li",
        },
        {
          name: "David Harwath",
        },
        {
          name: "Yu Tsao",
        },
        {
          name: "Shinji Watanabe",
        },
        {
          name: "Abdelrahman Mohamed",
        },
        {
          name: "Chi-Luen Feng",
        },
        {
          name: "Hung-yi Lee",
          link: "https://speech.ee.ntu.edu.tw/~hylee/index.php"
        },
      ],
      journal: "IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)",
      year: "2024",
      description: "We propose the AV-SUPERB benchmark that enables general-purpose evaluation of unimodal audio/visual and bimodal fusion representations on 7 datasets covering 5 audio-visual tasks in speech and audio processing.",
      others: {
        "Project Page": "https://av.superbbenchmark.org/",
        arXiv: "https://arxiv.org/abs/2309.10787",
        Code: "https://github.com/roger-tseng/av-superb"
      }
    },
    {
      image: DBC_img,
      title: "Diffusion Model-Augmented Behavioral Cloning",
      link: "https://nturobotlearninglab.github.io/dbc/",
      authors: [
        {
          name: "Hsiang-Chun Wang*",
          link: "https://hsiangchun0205.github.io/"
        },
        {
          name: "Shang-Fu Chen*",
          link: "https://shangfuchen.github.io/"
        },
        {
          name: "Ming-Hao Hsu",
          link: "https://qaz159qaz159.github.io/"
        },
        {
          name: "Chun-Mao Lai",
          link: "https://www.mecoli.net/",
          me: true,
        },
        {
          name: "Shao-Hua Sun",
          link: "https://shaohua0116.github.io/",
        },
      ],
      journal: "Frontiers4LCD Workshop at International Conference on Machine Learning (ICML)",
      year: "2023",
      description: "We propose a novel imitation learning method combining with diffusion model. We show that our method can achieve better performance than previous imitation learning methods.",
      others: {
        "Project Page": "https://nturobotlearninglab.github.io/dbc/",
        arXiv: "https://arxiv.org/abs/2302.13335",
      }
    },
    {
      image: CUDA_img,
      title: "Controllable User Dialogue Act Augmentation for Dialogue State Tracking",
      link: "https://arxiv.org/abs/2207.12757",
      authors: [
        {
          name: "Chun-Mao Lai*",
          link: "https://www.mecoli.net/",
          me: true,
        },
        {
          name: "Ming-Hao Hsu*",
          link: "https://qaz159qaz159.github.io/"
        },
        {
          name: "Chao-Wei Huang",
          link: "https://scholar.google.com/citations?user=nmsPLncAAAAJ",
        },
        {
          name: "Yun-Nung Chen",
          link: "https://www.csie.ntu.edu.tw/~yvchen/",
        },
      ],
      journal: "23rd Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL)",
      year: "2022",
      description: "We propose a data augmentation method for DST, which improve the state-of-the-art performance on MultiWOZ 2.1.",
      others: {
        arXiv: "https://arxiv.org/abs/2207.12757",
        Code: "https://github.com/miulab/cuda-dst"
      }
    }
  ]
}

export default researchData;