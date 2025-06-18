#include "mc_growth.h" 
#include "dlattice.h"
#include <algorithm>
#include <random>
#include <set>
#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

extern float AA_[Nconfig][6];
extern float BB[Nconfig][dir_number][9];
extern float transform_array[Nconfig][6];
void mc_step(void);   // шаг Монте-Карло
double calc_P_jump_sum(void); // сосчитать суммарную вероятность всех прыжков
void choose_jump(double P_jump_sum, int* x, int* y, int* z, int* dir); // выбрать прыжок
void do_many_axyz(void/*int x, int y, int z*/);  // сделать "шевеления" атомов после прыжка или осаждения

// Хелперы для индексации, чтобы не зависеть от cuda_kernels.cu
static inline void convert_host(int x, int y, int z, int Lx, int Ly, int &lx, int &ly, int &lz) {
    lx = x;
    ly = ((y >> 2) << 1) + ((y & 0x3) >> 1);
    lz = z;
}

static inline int get_atom_idx_host(int x, int y, int z, int Lx, int Ly, int Lz) {
    int lx, ly, lz;
    convert_host(x, y, z, Lx, Ly, lx, ly, lz);
    return (long)lz * (Ly / 2) * Lx + (long)ly * Lx + lx;
}


// void randn2_sequential_deterministic(float* x1, float* x2, const float* V) {
//   float S, f;
//   S = V[0] * V[0] + V[1] * V[1];
//   if (S >= 1.0f || S == 0.0f) { // Добавляем проверку как в оригинале
//       *x1 = 0.0f; *x2 = 0.0f; return;
//   }
//   f = sqrtf(-2.0f * logf(S) / S);
//   *x1 = V[0] * f;
//   *x2 = V[1] * f;
// }

// // Версия random_displacements, которая использует детерминированную randn2
// void random_displacements_sequential_deterministic(float* ax_, float* ay_, float* az_, unsigned short int config_, const float* randoms) {
//   // УБИРАЕМ ПРОВЕРКУ, чтобы соответствовать поведению оригинального кода
//   // if (config_ >= Nconfig || config_ == 65535 || param.T <= 0) { ... }

//   float a1, a2, a3, a4, coeff;
//   float* p;
//   coeff = sqrtf(0.5f * param.T); // Используем sqrtf для float

//   float V_set1[2] = {randoms[0], randoms[1]};
//   float V_set2[2] = {randoms[2], randoms[3]};
//   randn2_sequential_deterministic(&a1, &a2, V_set1);
//   randn2_sequential_deterministic(&a3, &a4, V_set2);
  
//   p = transform_array[config_];
//   *ax_ = coeff * (p[0] * a1 + p[3] * a2 + p[4] * a3);
//   *ay_ = coeff * (p[3] * a1 + p[1] * a2 + p[5] * a3);
//   *az_ = coeff * (p[4] * a1 + p[5] * a2 + p[2] * a3);
// }

// // Основная функция axyz из последовательной версии, адаптированная для теста
// float3 axyz_sequential(int x, int y, int z, const float* randoms) {
//   // Проверка на тип 0 была в оригинале, ее оставляем.
//   if (atoms(x,y,z).type == 0) return {0.0f, 0.0f, 0.0f};
  
//   // УБИРАЕМ ПРОВЕРКУ на config, чтобы соответствовать поведению оригинального кода
//   unsigned short int config_ = atoms(x,y,z).config;
//   // if (config_ >= Nconfig || config_ == 65535) return {0.0f, 0.0f, 0.0f};

//   // Используем float для всех вычислений, чтобы соответствовать GPU
//   float ax_, ay_, az_, ax2, ay2, az2, A_xx, A_yy, A_zz, A_xy, A_xz, A_yz;
//   float Bx, By, Bz, Bxx, Bxy, Bxz, Byx, Byy, Byz, Bzx, Bzy, Bzz;
//   int dir;
//   float* p;
  
//   p=AA_[config_]; 
//   A_xx=p[0]; A_yy=p[1]; A_zz=p[2]; A_xy=p[3]; A_xz=p[4]; A_yz=p[5];
  
//   Bx=atoms(x,y,z).B0.x;
//   By=atoms(x,y,z).B0.y;
//   Bz=atoms(x,y,z).B0.z;

//   neighbors_t nbs;
//   atoms.neighbors(x,y,z,nbs);

//   for (dir=0; dir<dir_number; dir++) {
//     ax2=atoms(nbs.x[dir],nbs.y[dir],nbs.z[dir]).a.x;
//     ay2=atoms(nbs.x[dir],nbs.y[dir],nbs.z[dir]).a.y;
//     az2=atoms(nbs.x[dir],nbs.y[dir],nbs.z[dir]).a.z;
//     p=BB[config_][dir];
//     Bxx=p[0]; Bxy=p[1]; Bxz=p[2]; Byx=p[3]; Byy=p[4]; Byz=p[5]; Bzx=p[6]; Bzy=p[7]; Bzz=p[8]; 
//     Bx+=Bxx*ax2+Bxy*ay2+Bxz*az2;
//     By+=Byx*ax2+Byy*ay2+Byz*az2;
//     Bz+=Bzx*ax2+Bzy*ay2+Bzz*az2;
//   }

//   ax_=ay_=az_=0.0f;
//   random_displacements_sequential_deterministic(&ax_,&ay_,&az_,config_, randoms);
//   ax_ -= 0.5f * ( A_xx*Bx+A_xy*By+A_xz*Bz );
//   ay_ -= 0.5f * ( A_xy*Bx+A_yy*By+A_yz*Bz );
//   az_ -= 0.5f * ( A_xz*Bx+A_yz*By+A_zz*Bz );
  
//   return {ax_, ay_, az_};
// }

// void run_comparison_test() {
//     std::cout << "--- Running Bulk Atom Comparison Test (70 atoms) ---" << std::endl;

//     if (spisok_atomov.empty()) {
//         std::cout << "ERROR: spisok_atomov is empty. Cannot run test." << std::endl;
//         return;
//     }

//     const int NUM_TEST_ATOMS = 70;
//     if (spisok_atomov.size() < NUM_TEST_ATOMS) {
//         std::cout << "ERROR: Not enough atoms in spisok_atomov to run test for " << NUM_TEST_ATOMS << " atoms." << std::endl;
//         return;
//     }

//     std::vector<axyz_work_item_t> work_items;
//     work_items.reserve(NUM_TEST_ATOMS);
    
//     // 1. Выбираем 70 случайных ВАЛИДНЫХ атомов для теста
//     srand(0); // Фиксируем seed для выбора атомов
//     while(work_items.size() < NUM_TEST_ATOMS) {
//         int n = random_(spisok_atomov.size());
//         coord c = spisok_atomov[n];
//         const auto& atom = atoms(c.x, c.y, c.z);

//         // Проверяем, что атом валидный (тип не 0)
//         if (atom.type != 0) {
//             axyz_work_item_t item;
//             item.center_coords = c;
//             item.center = atom;
//             work_items.push_back(item);
//         }
//     }
    
//     std::cout << "Selected " << work_items.size() << " atoms for testing." << std::endl;
    
//     // 2. Генерируем для каждого атома свой набор случайных чисел
//     // и заполняем данные о соседях
//     float t = 1.0f / (RAND_MAX + 1.0f);
//     neighbors_t nbs;
//     for(auto& item : work_items) {
//         for(int i=0; i<4; ++i) {
//             item.random_numbers[i] = 2.0f * t * (rand() + t * rand()) - 1.0f;
//         }
//         atoms.neighbors(item.center_coords.x, item.center_coords.y, item.center_coords.z, nbs);
//         for (int dir = 0; dir < dir_number; dir++) {
//             item.neighbor_coords[dir] = {nbs.x[dir], nbs.y[dir], nbs.z[dir]};
//             item.neighbors[dir] = atoms(nbs.x[dir], nbs.y[dir], nbs.z[dir]);
//         }
//     }

//     // 3. Запускаем последовательную версию для всех атомов
//     std::vector<float3> seq_results(NUM_TEST_ATOMS);
//     for(int i=0; i < NUM_TEST_ATOMS; ++i) {
//         seq_results[i] = axyz_sequential(
//             work_items[i].center_coords.x,
//             work_items[i].center_coords.y,
//             work_items[i].center_coords.z,
//             work_items[i].random_numbers
//         );
//     }

//     // 4. Запускаем параллельную версию для всех атомов
//     std::vector<axyz_result_t> par_results(NUM_TEST_ATOMS);
//     int ochered_count;
//     int max_ochered_size = NUM_TEST_ATOMS * (dir_number + 1);
//     std::vector<int> dummy_ochered(max_ochered_size * 3);

//     cuda_do_many_axyz_packed(
//         work_items.data(), par_results.data(), NUM_TEST_ATOMS, param.T, 
//         dummy_ochered.data(), dummy_ochered.data() + max_ochered_size, dummy_ochered.data() + 2*max_ochered_size, 
//         &ochered_count, max_ochered_size
//     );

//     // 5. Выводим результаты и считаем суммарную ошибку
//     std::cout << std::fixed << std::setprecision(8);
//     double total_diff = 0.0;
//     for(int i=0; i < NUM_TEST_ATOMS; ++i) {
//         const auto& seq_res = seq_results[i];
//         const auto& par_res = par_results[i].a;
//         double dx = fabs(seq_res.x - par_res.x);
//         double dy = fabs(seq_res.y - par_res.y);
//         double dz = fabs(seq_res.z - par_res.z);
//         total_diff += dx + dy + dz;

//         if (dx + dy + dz > 1e-6) { // Печатаем только если есть заметная разница
//             std::cout << "--- Mismatch found for atom " << i << " at (" 
//                       << work_items[i].center_coords.x << "," << work_items[i].center_coords.y << "," << work_items[i].center_coords.z 
//                       << ") config: " << work_items[i].center.config << " ---" << std::endl;
//             std::cout << "  CPU result: (" << seq_res.x << ", " << seq_res.y << ", " << seq_res.z << ")" << std::endl;
//             std::cout << "  GPU result: (" << par_res.x << ", " << par_res.y << ", " << par_res.z << ")" << std::endl;
//             std::cout << "  Difference: (" << dx << ", " << dy << ", " << dz << ")" << std::endl;
//         }
//     }

//     std::cout << "\n--- FINAL VERDICT ---" << std::endl;
//     std::cout << "Total absolute difference over " << NUM_TEST_ATOMS << " atoms: " << total_diff << std::endl;
//     if (total_diff < 1e-5) {
//         std::cout << "SUCCESS: The results are consistent." << std::endl;
//     } else {
//         std::cout << "FAILURE: Significant difference found between CPU and GPU results." << std::endl;
//     }

//     exit(0);
// }

void main_loop(void) { // бесконечный цикл, состоящий из шагов Монте-Карло
  int i, n , step = 0;
  int x, y, z;
  float ML_next_control;
  char filename[100];
  float ML_deposited;
  double t;
	float T;

  //printf("main_loop: starting, experiment_ML=%.2f\n", param.experiment_ML);
    ML_deposited = current.n_deposited / (Lx * Ly / 8.);
  //printf("main_loop: initial ML_deposited=%.2f, n_deposited=%d\n", ML_deposited, current.n_deposited);
  ML_next_control = param.dML_control * ((int) (ML_deposited / param.dML_control));// the last control check

  // если исходная структура без смещений и планируется моделирование роста, то 1000 раз пересчитываем смещения всех атомов
  if(((ML_deposited==0)&&param.experiment_ML!=0)){
    printf("main_loop: initializing displacements\n");
    for(n=0; n<1000; n++){
			std::random_shuffle(spisok_atomov.begin(),spisok_atomov.end());
      for(i=0;i<spisok_atomov.size();i++){
        struct coord c;
        c = spisok_atomov[i];
        axyz(c.x,c.y,c.z);
      }
    }
		ZXY v_ochered_Edef(x,y,z);
		update_Edef();
    cuda_init(&AA_[0][0], &BB[0][0][0], &transform_array[0][0], spisok_atomov.size(), param.moves_percent);
    //printf("main_loop: displacements initialized, spisok_atomov.size=%zu\n", spisok_atomov.size());
	}


  // Thermal annealing
  if (param.Edef_pa_N>0) {
  
	read_control("control.txt");
	float Edef_sum = 0;
	   
	FILE *f;

    for(n=0; n<param.Edef_pa_N; n++) {
	  
		std::random_shuffle(spisok_atomov.begin(),spisok_atomov.end());
		for (i=0; i<spisok_atomov.size(); i++) {
			struct coord c = spisok_atomov[i];
			axyz(c.x,c.y,c.z);
			calc_Edef_pa(c.x,c.y,c.z);
			Edef_sum += atoms(c.x,c.y,c.z).Edef_pa;
			atoms(c.x,c.y,c.z).Edef_pa = 0;
		}
	  
		sprintf(filename,"Esum_log.txt");
		f=fopen(filename,"at");
		if (f==NULL) {fprintf(stderr,"!!! cannot open log file '%s'\n",filename); return;}
		fprintf(f,"%d	%.2f	%.5f\n",n,param.T,Edef_sum); 
		fclose(f);
		Edef_sum = 0;	  
    }
	
	T = param.T;
	
	do {
		T -= 1.;
		if(T < 0) T = 0;
		
		for(n = 0; n < (T==0 ? 100 : 10); n++){
			
			std::random_shuffle(spisok_atomov.begin(),spisok_atomov.end());
			for (i=0; i<spisok_atomov.size(); i++) {
				struct coord c = spisok_atomov[i];
				axyz(c.x,c.y,c.z);
			}
		}

		for (i=0; i<spisok_atomov.size(); i++) {
			struct coord c = spisok_atomov[i];
			calc_Edef_pa(c.x,c.y,c.z);
			Edef_sum += atoms(c.x,c.y,c.z).Edef_pa;
			atoms(c.x,c.y,c.z).Edef_pa = 0;
		}

		sprintf(filename,"Esumcool_log.txt");
		f=fopen(filename,"at");
		if (f==NULL) {fprintf(stderr,"!!! cannot open log file '%s'\n",filename); return;}
		fprintf(f,"%.2f	%.5f\n",T,Edef_sum); 
		fclose(f);
		Edef_sum = 0;
	  

		
	} while(T > 0);
	
	sprintf(filename,"aaT0N%d.xyz",param.Edef_pa_N);
	show_me(filename,true);
    comp.notify(filename);
  }

	
	
	float show_short;
  float show_long;
  float show_sum;
	
	if(param.show_short != 0)	show_short = param.show_short * ((int) (ML_deposited / param.show_short));
	else show_short = 0;
	
	if(param.show_long != 0) show_long = param.show_long * ((int) (ML_deposited / param.show_long));
	else show_long = 0;
	
  if(param.show_sum != 0) show_sum = param.show_sum * ((int) (ML_deposited / param.show_sum));
	else show_sum = 0;
	
  t=current.t;
  
// main loop
  while (1) { // и запустим цикл
    //printf("main_loop: step=%d, ML_deposited=%.2f, t=%.2e, n_deposited=%d, spisok_atomov.size=%zu\n",step, ML_deposited, current.t, current.n_deposited, spisok_atomov.size());

  if (ML_deposited >= ML_next_control) {
    //printf("main_loop: writing log at ML_deposited=%.2f\n", ML_deposited);
      read_control("control.txt");
      write_to_log("log.txt");
      ML_next_control += param.dML_control;
    }

	if(param.show_short != 0)
		if(ML_deposited >= show_short) {
      //printf("main_loop: writing show_short at ML_deposited=%.2f\n", ML_deposited);
			sprintf(filename,"A%.2f.xyz",show_short);
			show_me(filename,0);
			comp.notify(filename);
			show_short += param.show_short;		
		}
	
	if(param.show_long != 0)
		if(ML_deposited >= show_long) {
      printf("main_loop: writing show_long at ML_deposited=%.2f\n", ML_deposited);
			sprintf(filename,"_A%.2f.xyz",show_long);
			show_me(filename,1);
			comp.notify(filename);
			show_long += param.show_long;		
		}
		
	if(param.show_sum != 0)
		if(ML_deposited >= show_sum) {
      //printf("main_loop: writing show_sum at ML_deposited=%.2f\n", ML_deposited);
			sprintf(filename,"__A%.2f.xyz",show_sum);
			show_me(filename,2);
			comp.notify(filename);
			show_sum += param.show_sum;
		}

  if (ML_deposited>=param.experiment_ML) {
    //printf("main_loop: experiment_ML reached, exiting\n");
    break;
  }
		
    mc_step();
    //printf("main_loop: after mc_step, n_deposited=%d, t=%.2e\n", current.n_deposited, current.t);
//	was 1 per 1e-6 s (mjus) at 670K
	if(current.t-t>=param.time_for_moves){	// calculate displacements per param.time_for_moves seconds
    //printf("main_loop: calling do_many_axyz\n");
    do_many_axyz();
	  t=current.t;
	}

/*
    i++;
    
    if (i%10000==0) {
      save_grid("M.txt",5);
      printf("step %d  t=%e  deposited=%d\n",i,current.t,current.n_deposited);
    }
    if (current.t>=t_next) {
      sprintf(filename,"M%.0f.txt",t_next);
      save_grid(filename,5);
      t_next+=dt;
    }
*/

  ML_deposited=current.n_deposited/(Lx*Ly/8.);
  if (step % 1000 == 0) {
    //printf("main_loop: checkpoint, step=%d, ML_deposited=%.2f\n", step, ML_deposited);
    }
  }
/*
  write_to_log("log.txt");
  if(param.experiment_ML!=0){
	sprintf(filename,"_A%.2f.xyz",ML_deposited);
	show_me(filename,1);
	comp.notify(filename);
  }
*/
//printf("main_loop: finished\n");

  // // Прямо в начале цикла вызываем наш тест
  //run_comparison_test(); 
}


void mc_step(void) { // шаг Монте-Карло
  double P_jump_sum,P_total,Px,dt;
  int x,y,z,dir,select,res,x2,y2,z2;
  //printf("mc_step: starting\n");
    update_Edef(); // приводим энергии деформ. и вероятности прыжков в соответствие с текущим положением
    //cuda_sync_atoms(atoms.lat, Lx, Ly, Lz);
    P_jump_sum=calc_P_jump_sum(); // сосчитаем суммарную вероятность всех прыжков
    //printf("mc_step: P_jump_sum=%.2e\n", P_jump_sum);
    
    P_total=P_jump_sum+param.p_deposition;
    Px=P_total*rand01();  // выберем, что делать - прыжок или осаждение

    //printf("mc_step: P_total=%.2e, Px=%.2e, p_deposition=%.2e\n", P_total, Px, param.p_deposition);
    if (Px<param.p_deposition) { // делаем осаждение
      select=1;
      //printf("mc_step: attempting deposition\n");
      res=deposition(param.dep_type,&x,&y,&z);
      //cudaMemcpy(dev_atoms, atoms.lat, Lx * Ly * Lz * sizeof(atom_t), cudaMemcpyHostToDevice);
      //printf("mc_step: deposition res=%d, x=%d, y=%d, z=%d\n", res, x, y, z);
    }
    else {                       // или делаем прыжок
      select=2;
      //printf("mc_step: attempting jump\n");
      //cuda_sync_atoms(atoms.lat, Lx, Ly, Lz);
      choose_jump(P_jump_sum,&x,&y,&z,&dir); // выберем, какой прыжок
      res=jump(x,y,z,dir,&x2,&y2,&z2);       // и выполним его
      //cudaMemcpy(dev_atoms, atoms.lat, Lx * Ly * Lz * sizeof(atom_t), cudaMemcpyHostToDevice);
      //printf("mc_step: jump res=%d, from (%d,%d,%d) dir=%d to (%d,%d,%d)\n", res, x, y, z, dir, x2, y2, z2);
    }
    //cuda_sync_atoms(atoms.lat, Lx, Ly, Lz);
    dt=-log(rand01())/P_total;   // обновим время
    current.t+=dt;

    //printf("mc_step: dt=%.2e, current.t=%.2e\n", dt, current.t);

    // сделаем "шевеления" атомов после прыжка или осаждения
//    if (select==1 && res==1) do_many_axyz(x,y,z); 
//    if (select==2 && res==1) do_many_axyz((x+x2)/2,(y+y2)/2,(z+z2)/2);
    
    // и напоследок обновим счётчики событий
    if (select==1 && res==1) current.n_deposited++;
    if (select==2 && res==0) current.n_bad_jumps++;
    if (select==2 && res==1) {
      current.n_jumps++;
      if (x2==current.prev_x && y2==current.prev_y && z2==current.prev_z) current.n_jumps_back++;
      current.prev_x=x; current.prev_y=y; current.prev_z=z;
    }
    //printf("mc_step: n_deposited=%d, n_jumps=%d, n_bad_jumps=%d\n",
      //current.n_deposited, current.n_jumps, current.n_bad_jumps);
    
//    if (select==1 && res==1) printf("                            %d   %d   %d.\n",x,y,z);
    //if (select==2 && res==0) printf("(%d %d %d) -> (%d %d %d) - otmenen\n",x,y,z,x2,y2,z2);
    //if (select==2 && res==1) printf("(%d %d %d) -> (%d %d %d)\n",x,y,z,x2,y2,z2);
    
}


double calc_P_jump_sum(void) { // сосчитать суммарную вероятность всех прыжков
  double real_sum = 0;
  int x, y, z;
  ZYX {
    real_sum += jump_probability(x,y,z);
  }
  return real_sum;
}


void choose_jump(double P_jump_sum, int* x_, int* y_, int* z_, int* dir_) { // выбрать прыжок
  int x,y,z,f,dir,n,nx;
  int j_array[dir_number];
  double P,Px;
  
  if ( unlikely(P_jump_sum<=0) ) {fprintf(stderr,"Error!!! in choose_jump(): P_jump_sum<=0\n"); exit(0);}
  
  Px=P_jump_sum*rand01();
  P=0; f=0;
  ZYX {
    P+=jump_probability(x,y,z);
    if (P>Px) {f=1; goto choose_jump_exit;} // нашли нужный узел (x,y,z) - выходим из цикла
  }
  choose_jump_exit: // выходим из цикла сюда
  
  if ( unlikely(f==0) )  {fprintf(stderr,"Error!!! in choose_jump() P=%f Px=%f P_jump_sum=%f\n",P,Px,P_jump_sum); exit(0);}
  if ( unlikely(x<0||x>=Lx||y<0||y>=Ly||z<2||z>=Lz-2) )  {fprintf(stderr,"Error!!! in choose_jump().\n"); exit(0);}
  if ( unlikely(n_jumps(x, y, z) <=0) ) {fprintf(stderr,"Error!!! in choose_jump()..\n"); exit(0);}
  
  fill_nb_type(j_array, jumps(x,y,z) );
  nx=1+random_( n_jumps(x,y,z) ); // которое по счёту направление прыжка мы выбираем
  n=0; f=0;
  for (dir=0; dir<dir_number; dir++) {
    n+=j_array[dir];
    if (n==nx) {f=1; break;}
  }
  
  if ( unlikely(f==0) )  {fprintf(stderr,"Error!!! in choose_jump()...\n"); exit(0);}
  if ( unlikely(dir<0 || dir>=dir_number) ) {fprintf(stderr,"Error!!! in choose_jump()....\n"); exit(0);}
  
  *x_=x; *y_=y; *z_=z; *dir_=dir;
}


/*
void do_many_axyz(int x0, int y0, int z0) {  // сделать "шевеления" атомов после прыжка или осаждения
  int x,y,z, i,i_, I2,Imax;
  I2=500; //1000;
  Imax=1000000;
  i=0;
  for (i_=0; i_<Imax; i_++) {
    z = 2 + random_(Lz-4);
    x = (z%2) + 2*random_(Lx/2);
    y = (z%2) + 2*((z/2+x/2)%2) + 4*random_(Ly/4);
    if (x<0 || x>=Lx || y<0 || y>=Ly || z<2 || z>=Lz-2)
      {fprintf(stderr,"Error!!! in do_many_axyz()\n"); exit(0);}
    if (atom_type[x][y][z]!=0) {
      axyz(x,y,z);
      i++;
    }
    if (i>=I2) break;
  }
}
*/


// void do_many_axyz(void/*int x0, int y0, int z0*/) {  // сделать "шевеления" атомов после прыжка или осаждения
//   int i,I,n;
//   I=(int) (param.moves_percent/100.*spisok_atomov.size());
//   for (i=0; i<I; i++) {
//     n=random_(spisok_atomov.size());
//     struct coord c = spisok_atomov[n];
//             bbbbbbb(c.x,c.y,c.z);
// 	current.n_moves++;
//   }
// }
void do_many_axyz(void) {
  int I = (int)(param.moves_percent / 100. * spisok_atomov.size());
  if (I <= 0) {
    return;
  }
  if (spisok_atomov.empty()) {
    return;
  }

  // Создаем векторы для "пакетов"
  std::vector<axyz_work_item_t> work_items(I);
  std::vector<axyz_result_t> results(I);
  std::vector<int> global_indices(I);

  neighbors_t nbs; // Структура для получения соседей

  for (int i = 0; i < I; i++) {
      struct coord c;
      do {
          int n = random_(spisok_atomov.size());
          c = spisok_atomov[n];
      } while (atoms(c.x, c.y, c.z).type == 0);
      
      global_indices[i] = get_atom_idx_host(c.x, c.y, c.z, Lx, Ly, Lz);
      work_items[i].center_coords = c;
      work_items[i].center = atoms(c.x, c.y, c.z);

      atoms.neighbors(c.x, c.y, c.z, nbs); // Получаем соседей

      for (int dir = 0; dir < dir_number; dir++) {
          const int& x2 = nbs.x[dir];
          const int& y2 = nbs.y[dir];
          const int& z2 = nbs.z[dir];
          work_items[i].neighbor_coords[dir] = {x2, y2, z2};
          work_items[i].neighbors[dir] = atoms(x2, y2, z2);
      }
  }

  // Готовим буферы для очереди
  int max_ochered_size = I * (dir_number + 1);
  std::vector<int> host_ochered_x(max_ochered_size);
  std::vector<int> host_ochered_y(max_ochered_size);
  std::vector<int> host_ochered_z(max_ochered_size);
  int ochered_count = 0;

  // Вызываем "пакетную" CUDA-функцию
  cuda_do_many_axyz_packed(
      work_items.data(), results.data(), I, param.T,
      host_ochered_x.data(), host_ochered_y.data(), host_ochered_z.data(),
      &ochered_count, max_ochered_size
  );

  // Update atoms on CPU with results from GPU
  for (int i = 0; i < I; i++) {
      coord c = work_items[i].center_coords;
      atoms(c.x, c.y, c.z).a.x = results[i].a.x;
      atoms(c.x, c.y, c.z).a.y = results[i].a.y;
      atoms(c.x, c.y, c.z).a.z = results[i].a.z;
  }
  
  // Добавляем атомы и их соседей в очередь на обновление энергии
  for (int j = 0; j < ochered_count; j++) {
   //printf("Count of ochered: %d\n",ochered_count);
      v_ochered_Edef(host_ochered_x[j], host_ochered_y[j], host_ochered_z[j]);
  }
  current.n_moves += I;

}