from typing import Callable, Union, Tuple, Optional

from numpy import ndarray

from ..core.instance import InstanceBase


class ParticleMeta(object):
    def __init__(self, vel: ndarray, c1: float, c2: float, w: Optional[float] = None) -> None:
        self.vel = vel
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.fitness = None
        self.best_fitness = None
        self.best_pos = None
        
    def __str__(self):
        self_inf_st = f"""
            Velocity:       {self.vel}
            c1:             {self.c1}
            c2:             {self.c2}
            w:              {self.w}
            Fitness:        {self.fitness}
            Best Fitness:   {self.best_fitness}
            Best Position:  {self.best_pos}
        """
        return '{\n' + self_inf_st + '\n}'
    
    def copy(self):
        return ParticleMeta(
            self.vel.copy(),
            self.c1,
            self.c2,
            self.w,
            self.fitness,
            self.best_fitness,
            self.best_pos.copy()
        )
    


class ParticleInstance(InstanceBase):
    def __init__(
        self,
        pos_initializer: Union[ndarray, Callable[[], ndarray]],
        vel_initializer: Union[ndarray, Callable[[], ndarray]],
        c: Union[float, Tuple[float, float], Callable[[], float], Callable[[], Tuple[float, float]]],
        w: Optional[Union[float, Callable[[], float]]] = None
    ) -> None:
        def _initialize():    
            meta = ParticleMeta(vel, c1, c2, w_)
            return solution, meta
        
        c2 = None
        if isinstance(c, float):
            c1 = c
        elif isinstance(c, tuple):
            c1, c2 = c
        elif isinstance(c, Callable):
            cs = c()
            if isinstance(cs, tuple):
                c1 = cs
            else:
                c1, c2 = cs
        
        if c2 is None:
            c2 = 4.0 - c1

        if c1 < 0.0:
            raise ValueError(f"c1 must be nonnegative (found {c1})")
        if c2 < 0.0:
            raise ValueError(f"c1 must be nonnegative (found {c2})")
        
        if isinstance(w, Callable):
            w_ = w()
        else:
            w_ = w
            
        if isinstance(pos_initializer, Callable):
            solution = pos_initializer()
        else:
            solution = pos_initializer
            
        if isinstance(vel_initializer, Callable):
            vel = vel_initializer()
        else:
            vel = vel_initializer

        super(ParticleInstance, self).__init__(_initialize)

    def __str__(self):
        solution_st = str(self.solution)
        meta_st = str(self.meta)
        return f"""
        Particle(
            Solution: {solution_st},
            Meta: {meta_st}
        )
        """

    def copy(self):
        out = ParticleInstance(
            self.solution.copy(),
            self.meta.vel.copy(),
            (self.meta.c1, self.meta.c2),
            self.meta.w
        )
        out.meta.fitness = self.meta.fitness
        out.meta.best_fitness = self.meta.best_fitness
        out.meta.best_pos = self.meta.best_pos.copy()
        
        return out