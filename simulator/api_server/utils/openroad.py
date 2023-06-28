import multiprocessing
import os

import drt

from openroad import Design, Tech, set_thread_count


def detailed_route(design, *,
                   output_maze="",
                   output_drc="",
                   output_cmap="",
                   output_guide_coverage="",
                   db_process_node="",
                   disable_via_gen=False,
                   droute_end_iter=-1,
                   via_in_pin_bottom_layer="",
                   via_in_pin_top_layer="",
                   or_seed=-1,
                   or_k=0,
                   bottom_routing_layer="",
                   top_routing_layer="",
                   verbose=0,
                   clean_patches=False,
                   no_pin_access=False,
                   single_step_dr=False,
                   min_access_points=-1,
                   save_guide_updates=False):
    params = drt.ParamStruct()
    params.outputMazeFile = output_maze
    params.outputDrcFile = output_drc
    params.outputCmapFile = output_cmap
    params.outputGuideCoverageFile = output_guide_coverage
    params.dbProcessNode = db_process_node
    params.enableViaGen = not disable_via_gen
    params.drouteEndIter = droute_end_iter
    params.viaInPinBottomLayer = via_in_pin_bottom_layer
    params.viaInPinTopLayer = via_in_pin_top_layer
    params.orSeed = or_seed
    params.orK = or_k
    params.bottomRoutingLayer = bottom_routing_layer
    params.topRoutingLayer = top_routing_layer
    params.verbose = verbose
    params.cleanPatches = clean_patches
    params.doPa = not no_pin_access
    params.singleStepDR = single_step_dr
    params.minAccessPoints = min_access_points
    params.saveGuideUpdates = save_guide_updates

    router = design.getTritonRoute()
    router.setParams(params)
    router.main()


def create_task(lef_file,
                def_file,
                guide_file,
                output_drc=None,
                output_maze=None,
                output_def=None,
                droute_end_iter=-1,
                verbose=0):
    tech = Tech()
    tech.readLef(lef_file)
    design = Design(tech)
    design.readDef(def_file)
    gr = design.getGlobalRouter()
    gr.readGuides(guide_file)

    if output_drc is not None:
        dir = output_drc.split(os.sep)[:-1]
        dir = os.sep.join(dir)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

    if output_maze is not None:
        dir = output_maze.split(os.sep)[:-1]
        dir = os.sep.join(dir)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

    set_thread_count(multiprocessing.cpu_count())

    detailed_route(design,
                   output_drc=output_drc,
                   output_maze=output_maze,
                   droute_end_iter=droute_end_iter,
                   verbose=verbose)

    # write def file
    if output_def is not None:
        dir = output_def.split(os.sep)[:-1]
        dir = os.sep.join(dir)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        design.writeDef(output_def)


def create_task_by_testcase_name(name, droute_end_iter=-1, verbose=0):
    lef_file = os.path.join(os.sep, 'app', 'testcases', name, f'{name}.input.lef')
    def_file = os.path.join(os.sep, 'app', 'testcases', name, f'{name}.input.def')
    guide_file = os.path.join(os.sep, 'app', 'testcases', name, f'{name}.input.guide')
    output_drc = os.path.join(os.sep, 'app', 'results', name, f'{name}.output.drc.rpt')
    output_maze = os.path.join(os.sep, 'app', 'results', name, f'{name}.output.maze.log')
    output_def = os.path.join(os.sep, 'app', 'results', name, f'{name}.output.def')

    create_task(lef_file,
                def_file,
                guide_file,
                output_drc,
                output_maze,
                output_def,
                droute_end_iter=droute_end_iter,
                verbose=verbose)
