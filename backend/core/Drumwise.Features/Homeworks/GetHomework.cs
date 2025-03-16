using AutoMapper;
using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Mappings;
using Drumwise.Application.Common.Models;
using Drumwise.Application.Entities;
using Drumwise.Infrastructure.Data;
using Drumwise.Infrastructure.Identity.Constants;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace Drumwise.Features.Homeworks;

public record GetHomeworkQuery(string HomeworkId) : IRequest<(Result, HomeworkItemDto?)>;


public class GetHomeworkHandler(ApplicationDbContext context, IMapper mapper, IUser user)
    : IRequestHandler<GetHomeworkQuery, (Result, HomeworkItemDto?)>
{
    public async Task<(Result, HomeworkItemDto?)> Handle(GetHomeworkQuery request, CancellationToken cancellationToken)
    {
        var loggedUser = user.Id;
        var ifTeacher = await user.IsInRoleAsync(Roles.Teacher);
        
        var homeworkRaw = ifTeacher
            ? await context.Homeworks
                .FirstOrDefaultAsync(x => x.Id == Guid.Parse(request.HomeworkId)
                                                                  && x.CreatedBy == loggedUser, cancellationToken).ConfigureAwait(false)
            : await context.Homeworks
                .FirstOrDefaultAsync(x => x.Id == Guid.Parse(request.HomeworkId)
                                          && x.AssignedTo == loggedUser, cancellationToken).ConfigureAwait(false);

        if (homeworkRaw is null)
            return (Result.Failure(HomeworkErrors.HomeworkNotFound, ResultType.NotFound), null);

        var homeworkDto = mapper.Map<HomeworkItemDto>(homeworkRaw);

        return (Result.Success(ResultType.Ok), homeworkDto);
    }
}

public record HomeworkItemDto : IMapFrom<Homework>
{
    public Guid Id { get; set; }
    public DateTime Deadline { get; set; }
    public required string HomeworkTitle { get; set; }
    public required string AssignedTo { get; set; }
    public required string CreatedBy { get; set; }
    public required string Exercise { get; set; }
}